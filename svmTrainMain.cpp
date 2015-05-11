#include <stdio.h>
#include <stdlib.h>
#include "svmTrainMain.hpp"
#include "svmTrain.h"
#include "parse.hpp"
#include <iostream>
#include <vector>
#include <iterator>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <vector>
#include "CycleTimer.h"
#include <unistd.h>
#include <mpi.h>
#include <fstream>

state_model state;

using namespace std;

static void usage_exit() {
    cerr <<
"   Command Line:\n"
"\n"
"   -a/--num-att        :  [REQUIRED] The number of attributes\n"
"                                     /features\n"
"   -x/--num-ex         :  [REQUIRED] The number of training \n"
"                                     examples\n"
"   -f/--file-path      :  [REQUIRED] Path to the training file\n"
"   -c/--cost           :  Parameter c of the SVM (default 1)\n"
"   -g/--gamma          :  Parameter gamma of the radial basis\n"
"                          function: exp(-gamma*|u-v|^2)\n"
"                          (default: 1/num-att)\n"
"   -e/--epsilon        :  Tolerance of termination criterion\n"
"                          (default 0.001)\n"
"   -n/--max-iter       :  Maximum number of iterations\n"
"                          (default 150,000\n"
"   -m/--model          :  [REQUIRED] Path of model to be saved\n"
"   -s/--cache-size     :  Size of cache (num cache lines)\n"
"\n";
    
	exit(-1);
}

static struct option longOptionsG[] =
{
    { "num-att",        required_argument,          0,  'a' },
    { "num-ex",         required_argument,          0,  'x' },
    { "cost",           required_argument,          0,  'c' },
    { "gamma",          required_argument,          0,  'g' },
    { "file-path",      required_argument,          0,  'f' },
    { "epsilon",       	required_argument,          0,  'e' },
    { "max-iter",		required_argument,			0,	'n'	},
    { "model",			required_argument,			0,	'm' },
    { "cache-size",		required_argument,			0,	's' },
    { 0,                0,                          0,   0  }
};

static void parse_arguments(int argc, char* argv[]) {

    // Default Values
    state.epsilon = 0.001;
    state.c = 1;
	state.num_attributes = -1;
	state.num_train_data = -1;
	state.gamma = -1;
	strcpy(state.input_file_name, "");
	strcpy(state.model_file_name, "");
	state.max_iter = 150000;
	state.cache_size = 10;

    // Parse args
    while (1) {
        int idx = 0;
        int c = getopt_long(argc, argv, "a:x:c:g:f:e:n:m:s:", longOptionsG, &idx);

        if (c == -1) {
            // End of options
            break;
        }

        switch (c) {
        case 'a':
            state.num_attributes = atoi(optarg);
            break;
        case 'x':
            state.num_train_data = atoi(optarg);
            break;
        case 'c':
            state.c = atof(optarg);
            break;
        case 'g':
            state.gamma = atof(optarg);
            break;
       	case 'f':
            strcpy(state.input_file_name, optarg);
            break;
       	case 'e':
            state.epsilon = atof(optarg);
            break;
       	case 'n':
       		state.max_iter = atoi(optarg);
       		break;
       	case 'm':
       		strcpy(state.model_file_name, optarg);
       		break;
       	case 's':
       		state.cache_size = atoi(optarg);
       		break;
        default:
            cerr << "\nERROR: Unknown option: -" << c << "\n";
            // Usage exit
            usage_exit();
        }
    }

	if(strcmp(state.input_file_name,"")==0 || strcmp(state.model_file_name,"")==0) {

		cerr << "Enter a valid file name\n";
		usage_exit();
	}

	if(state.num_attributes <= 0 || state.num_train_data <= 0) {

		cerr << "Missing a required parameter, or invalid parameter\n";
		usage_exit();

	}

	if(state.gamma < 0) {

		state.gamma = 1 / state.num_attributes;
	}

}

void initialize_shard_sizes(int *shard_size, int cluster_size);
void initialize_shard_disp(int *shard_size, int *shard_disp, int cluster_size);
void write_out_model(float* x, int* y, float* alpha, float b);

int main(int argc, char *argv[]) {

	MPI::Init(argc, argv);

	//Obtain the command line arguments
	parse_arguments(argc, argv);

	//MPI statistics
	int rank, cluster_size;

	//Open MPI initialization
	cluster_size = MPI::COMM_WORLD.Get_size();
	rank = MPI::COMM_WORLD.Get_rank();
	
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int len;
	memset(hostname,0,MPI_MAX_PROCESSOR_NAME);
	MPI::Get_processor_name(hostname,len);
	memset(hostname+len,0,MPI_MAX_PROCESSOR_NAME-len);

	//distribution of datapoints in each shard
	int *shard_size;
	int *shard_disp;

	//input data attributes and labels
	std::vector<float> raw_x(state.num_train_data * state.num_attributes,0);
	std::vector<int> raw_y(state.num_train_data,0);

	//initialize shard sizes
	shard_size = new int[cluster_size];
	initialize_shard_sizes(shard_size, cluster_size);

	//initialize shard displacements
	shard_disp = new int[cluster_size];
	initialize_shard_disp(shard_size, shard_disp, cluster_size);

	//cout << "cluster_size: " << cluster_size << "\trank: " << rank << "\n";

	populate_data(raw_x, raw_y, state.num_train_data, state.num_attributes, state.input_file_name);
	cout << "Populated Data from input file at node: " << rank << '\n';

	MPI::COMM_WORLD.Barrier();

	if(rank == 0) {
		for(int i=0; i<cluster_size; i++) {
			cout << shard_disp[i] << '\t' << shard_size[i] << '\n';
		}
	}

	MPI::COMM_WORLD.Barrier();

	//SVM class initialization (locl to every process)
	SvmTrain svm(shard_size[rank], shard_disp[rank]);
		
	svm.setup(raw_x, raw_y);

	MPI::COMM_WORLD.Barrier();

	if(rank == 0) {

		cout << "SETUP DONE\n";

	}
	//timer for training
	unsigned long long start = 0;
	if(rank == 0) {
		start = CycleTimer::currentSeconds();
	}

	float b_lo = 0;
	float b_hi = 0;

	//check num iter for convergence
	int num_iter = 0;

	//gather variables from local processes
	int *I_lo, *I_hi;
	float *f_hi, *f_lo;
	float *recv;

	//alpha copies at root node, initialized to zero
	float *alpha = new float[state.num_train_data]();

	//assign receive buffers at root
	I_lo = new int[cluster_size];
	I_hi = new int[cluster_size];
	f_lo = new float[cluster_size];
	f_hi = new float[cluster_size];

	recv = new float[4*cluster_size];

	MPI::COMM_WORLD.Barrier();

	do {

		//if(rank == 0) {
		//	cout << "Iteration: " << num_iter << "\n";	
		//}

		svm.train_step1();

		//gather all local extremes at every node
		MPI::COMM_WORLD.Allgather(svm.rv, 4, MPI_FLOAT, recv, 4, MPI_FLOAT);

		int I_lo_global, I_hi_global;
		float alpha_lo_new, alpha_hi_new;
			
		float max = -1000000000;
		float min = 1000000000;
		int max_idx = 0;
		int min_idx = 0;

		//convert gathered array to separate arrays
		for(int i=0; i<cluster_size; i++) {
			int idx = i*4;
			I_hi[i] = (int)recv[idx];
			I_lo[i] = (int)recv[idx+1];
			f_hi[i] = recv[idx+2];
			f_lo[i] = recv[idx+3];
		}

		//obtain global maximas
		for(int i=0; i<cluster_size; i++) {
			if(f_lo[i] > max) {
				max = f_lo[i];
				max_idx = I_lo[i];
			}

			if(f_hi[i] < min) {
				min = f_hi[i];
				min_idx = I_hi[i];
			}
		}

		b_lo = max;
		b_hi = min;

		int y_lo = raw_y[max_idx];
		int y_hi = raw_y[min_idx];

		float eta = svm.rbf_kernel(min_idx,min_idx) + svm.rbf_kernel(max_idx,max_idx) - (2*svm.rbf_kernel(max_idx,min_idx));

		//obtain alpha_low and alpha_hi (old values)
		float alpha_lo_old = alpha[max_idx];
		float alpha_hi_old = alpha[min_idx];

		//update alpha_low and alpha_hi
		float s = y_lo*y_hi;
		alpha_lo_new = alpha_lo_old + (y_lo*(b_hi - b_lo)/eta);
		alpha_hi_new = alpha_hi_old + (s*(alpha_lo_old - alpha_lo_new));

		//clip new alpha values between 0 and C
		alpha_lo_new = svm.clip_value(alpha_lo_new, 0.0, state.c);
		alpha_hi_new = svm.clip_value(alpha_hi_new, 0.0, state.c);

		//store new alpha_lo and alpha_hi values at root
		alpha[max_idx] = alpha_lo_new;
		alpha[min_idx] = alpha_hi_new;

		I_lo_global = max_idx;
		I_hi_global = min_idx;
		
		//step2 of svm training iteration
		svm.train_step2(I_hi_global, I_lo_global, alpha_hi_new, alpha_lo_new);

		//reach convergence
		num_iter++;

	} while((b_lo > (b_hi +(2*state.epsilon))) && num_iter < state.max_iter);
	
	unsigned long long t2 = CycleTimer::currentSeconds();
	if(rank == 0) {
		cout << "TOTAL TIME TAKEN in seconds: " << t2-start << "\n";

		//check if converged or max_iter stop
		if(b_lo > (b_hi + (2*state.epsilon))) {
			cout << "Could not converge in " << num_iter << " iterations. SVM training has been stopped\n";
		} else {
			cout << "Converged at iteration number: " << num_iter << "\n";
		}

	}

	svm.destroy_cuda_handles();

	if(rank == 0) {
		//obtain final b intercept
		svm.b = (b_lo + b_hi)/2;
		cout << "b: " << svm.b << "\n";
	
		svm.test_setup();

		//obtain training accuracy
		float train_accuracy = svm.get_train_accuracy();
		cout << "Training accuracy: " << train_accuracy << "\n";
	
		svm.destroy_t_cuda_handles();
	}

	//write training model
	if(rank == 0) {
		//refernce vectors as arrays
		float *x_arr = &raw_x[0];
		int *y_arr = &raw_y[0];

		//write model to file
		write_out_model(x_arr, y_arr, alpha, svm.b);

		cout << "Training model has been saved to the file " << state.model_file_name << "\n";
	}

	//clear training data
	raw_x.clear();
	raw_y.clear();

	raw_x.shrink_to_fit();
	raw_y.shrink_to_fit();

	delete [] alpha;

	MPI::Finalize();

	return 0;
}

void initialize_shard_sizes(int *shard_size, int cluster_size) {
	int num_data_shard = (int)ceil((double)state.num_train_data/(double)cluster_size);

	for(int i=0; i<cluster_size-1; i++) {
		shard_size[i] = num_data_shard;
	}

	int num_data_last_shard = state.num_train_data - ((cluster_size-1)*num_data_shard);
	shard_size[cluster_size-1] = num_data_last_shard;
}

void initialize_shard_disp(int *shard_size, int *shard_disp, int cluster_size) {
	shard_disp[0] = 0;

	for(int i=1; i<cluster_size; i++) {
		shard_disp[i] = shard_disp[i-1] + shard_size[i-1];
	}
}

void write_out_model(float* x, int* y, float* alpha, float b) {
	//open output filestream for writing the model
	ofstream model_file;
	model_file.open(state.model_file_name);

	if(model_file.is_open()) {
		//gamma used in kernel for training
		model_file << state.gamma << "\n";
		model_file << b << "\n";

		for(int i=0; i<state.num_train_data; i++) {
			if(alpha[i] != 0) {
				model_file << alpha[i] << "," << y[i];

				//index for x
				int idx = i*(state.num_attributes);

				for(int j=0; j<state.num_attributes; j++) {
					model_file << "," << x[idx + j];
				}

				model_file << "\n";
			}
		}

		model_file.close();
	} else {
		cout << "Model output file " << state.model_file_name << " could not be opened for writing.\n";
		exit(-1);
	}
}