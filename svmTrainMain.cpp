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

state_model state;

using namespace std;

static void usage_exit() {
    cerr <<
"   Command Line:\n"
"\n"
"   -a/--num-att        :  [REQUIRED] The number of attributes\n"
"									  /features\n"
"   -x/--num-ex       	:  [REQUIRED] The number of training \n"
"									  examples\n"
"   -f/--file-path      :  [REQUIRED] Path to the training file\n"
"   -c/--cost        	:  Parameter c of the SVM (default 1)\n"
"   -g/--gamma       	:  Parameter gamma of the radial basis\n"
"						   function: exp(-gamma*|u-v|^2)\n"
"						   (default: 1/num-att)"
"   -e/--epsilon        :  Tolerance of termination criterion\n"
"						   (default 0.001)"
"	-n/--max-iter		:  Maximum number of iterations\n"
"						   (default 150,000"
"	-m/--model 			:  [REQUIRED] Path of model to be saved\n"
"	-s/--cache-size		:  Size of cache (num cache lines)\n"
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

	//MPI::COMM_WORLD.Scatterv(raw_x_array, shard_size_x, shard_disp_x, MPI_FLOAT, raw_x_shard, shard_size_x[rank], MPI_FLOAT, 0);
	//MPI::COMM_WORLD.Scatterv(raw_y_array, shard_size_y, shard_disp_y, MPI_INT, raw_y_shard, shard_size_y[rank], MPI_INT, 0);

	//cout << "Scatter complete at node " << rank << "\n";

	MPI::COMM_WORLD.Barrier();

	for(int i = 0; i < cluster_size; i++) {
		MPI::COMM_WORLD.Barrier();
		if (i == rank) {
    		for(int j=0; j<state.num_attributes; j++) {
				cout << raw_x[j] << ',';
			}
			cout << "\n\n";
		}
	}

	/*
	//convert shard data to vectors
	raw_x.assign(raw_x_shard, raw_x_shard + shard_size_x[rank]);
	raw_y.assign(raw_y_shard, raw_y_shard + shard_size_y[rank]);

	//SVM class initialization (locl to every process)
	SvmTrain svm(shard_size_y[rank], shard_disp_y[rank]);
	svm.setup(raw_x, raw_y);

	MPI::COMM_WORLD.Barrier();

	//timer for training
	unsigned long long start;
	if(rank == 0) {
		start = CycleTimer::currentSeconds();
	}

	int num_iter = 0;

	do {

		svm.train_step();		

		num_iter++;

		//	cout << "--------------------------------\n";

	} while((svm.b_lo > (svm.b_hi +(2*state.epsilon))) && num_iter < state.max_iter);
	
	unsigned long long t2 = CycleTimer::currentSeconds();
	cout << "TOTAL TIME TAKEN in seconds: " << t2-start << "\n";

	//check if converged or max_iter stop
	if(svm.b_lo > (svm.b_hi + (2*state.epsilon))) {
		cout << "Could not converge in " << num_iter << " iterations. SVM training has been stopped\n";
	} else {
		cout << "Converged at iteration number: " << num_iter << "\n";
	}

	svm.destroy_cuda_handles();

	//obtain final b intercept
	svm.b = (svm.b_lo + svm.b_hi)/2;
	cout << "b: " << svm.b << "\n";

	//obtain training accuracy
	float train_accuracy = svm.get_train_accuracy();
	cout << "Training accuracy: " << train_accuracy << "\n";

	//write model to file
	//write_out_model(x, y, alpha, b);

	//cout << "Training model has been saved to the file " << state.model_file_name << "\n";

	//clear training data
	//for(int i = 0 ; i < state.num_train_data; i++) {	
	//	delete [] x[i];
	//}

	//delete [] x;
	//delete [] y;
	*/

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