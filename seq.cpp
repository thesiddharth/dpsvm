#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <cblas.h>
#include <math.h>
#include <getopt.h>
#include <string>
#include "CycleTimer.h"

using namespace std;

void populate_data(float** x, int* y);
void initialize_f_array(float* f, int* y);
void set_I_arrays(float* alpha, int *y, vector<int> I[5]);
int get_I_up(float* f, vector<int> I[5]);
int get_I_low(float* f, vector<int> I[5]);
void get_x(float* x, float* x_copy, int idx, int num_attributes);
float rbf_kernel(float* x1,float* x2);
float clip_value(float num, float low, float high);
void update_f(float* f, float** x, int I_low, int I_hi, int y_low, int y_hi, float alpha_low_old, float alpha_hi_old, float alpha_low_new, float alpha_hi_new);
float get_duality_gap(float* alpha, int* y, float* f, float c, float b, int num_train_data, int num_attributes);
void print_x(float* x);
float get_train_accuracy(float** x, int* y, float* alpha, float b);
void write_out_model(float** x, int* y, float* alpha, float b);

typedef struct {

	int num_attributes;
	int num_train_data;
	float c;
	float gamma;
	float epsilon;
	char input_file_name[60];
	char model_file_name[60];
	int max_iter;

} state_model;

//global structure for training parameters
static state_model state;

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

    // Parse args
    while (1) {
        int idx = 0;
        int c = getopt_long(argc, argv, "a:x:c:g:f:e:n:m:", longOptionsG, &idx);

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

int main(int argc, char *argv[]) {

	//Obtain the command line arguments
	parse_arguments(argc, argv);

	//input data attributes and labels
	float** x = new float*[state.num_train_data];
	for(int i = 0 ; i < state.num_train_data; i++) {
		
		x[i] = new float[state.num_attributes]();

	}

	int* y = new int[state.num_train_data];

	//read data from input file
	populate_data(x, y);

	cout << "Populated Data from input file\n";
	
	unsigned long long t2, start;
	start = CycleTimer::currentSeconds();

	//Initialize starting values
	float* alpha = new float[state.num_train_data]();
	float* f = new float[state.num_train_data];

	initialize_f_array(f, y);

	//I sets based on alpha and y
	vector<int> I[5];

	//b (intercept), checks optimality condition for stopping
	float b_low, b_hi;

	//check iteration number for stopping condition
	int num_iter = 0;

	do {

		//update the I sets
		set_I_arrays(alpha, y, I);

		//get b_hi and b_low
		int I_hi = get_I_up(f,I);
		b_hi = f[I_hi];
		
		int I_low = get_I_low(f,I);
		b_low = f[I_low];
		
		/*if(num_iter == 100) {
			
			cout << "-----\n";

			for(vector<int>::iterator it = I[4].begin(); it != I[4].end(); ++it) {
				
				 cout << *it << ": " << f[*it] << ",\n";

			}	

			cout << "\n-----\n";

			exit(0);
		}*/

		int y_low = y[I_low];
		int y_hi = y[I_hi];

// 	    cout << "I_lo: \t" << I_low << ", I_hi: \t" << I_hi << '\n';
// 	    cout << "b_lo: \t" << b_low << ", b_hi: \t" << b_hi << '\n';
		
		float eta = rbf_kernel(x[I_hi],x[I_hi]) + rbf_kernel(x[I_low],x[I_low]) - (2*rbf_kernel(x[I_low],x[I_hi])) ;

		//obtain alpha_low and alpha_hi (old values)
		float alpha_low_old = alpha[I_low];
		float alpha_hi_old = alpha[I_hi];

//        cout << "eta: " << eta << '\n';
	
		//update alpha_low and alpha_hi
		float s = y_low*y_hi;
		float alpha_low_new = alpha_low_old + (y_low*(b_hi - b_low)/eta);
		float alpha_hi_new = alpha_hi_old + (s*(alpha_low_old - alpha_low_new));

		//clip new alpha values between 0 and C
		alpha_low_new = clip_value(alpha_low_new, 0.0, state.c);
		alpha_hi_new = clip_value(alpha_hi_new, 0.0, state.c);
		
//        cout << "alpha_lo_new: " << alpha_low_new << '\n';
//        cout << "alpha_hi_new: " << alpha_hi_new << '\n';
	
		//store new alpha_1 and alpha_2 values
		alpha[I_low] = alpha_low_new;
		alpha[I_hi] = alpha_hi_new;

		//update f values
		update_f(f, x, I_low, I_hi, y_low, y_hi, alpha_low_old, alpha_hi_old, alpha_low_new, alpha_hi_new);

		//Increment number of iterations to reach stopping condition
		num_iter++;

		cout << "Current iteration number: " << num_iter << "\n";

	} while((b_low > (b_hi +(2*state.epsilon))) && num_iter < state.max_iter);
	
	t2 = CycleTimer::currentSeconds();
	cout << "TOTAL TIME TAKEN in seconds: " << t2-start << "\n";

	if(b_low > (b_hi + (2*state.epsilon))) {
		cout << "Could not converge in " << num_iter << " iterations. SVM training has been stopped\n";
	} else {
		cout << "Converged at iteration number: " << num_iter << "\n";
	}

	//obtain final b intercept
	float b = (b_low + b_hi)/2;
	cout << "b: " << b << "\n";

	//obtain training accuracy
	float train_accuracy = get_train_accuracy(x, y, alpha, b);
	cout << "Training accuracy: " << train_accuracy << "\n";

	//write model to file
	write_out_model(x, y, alpha, b);

	cout << "Training model has been saved to the file " << state.model_file_name << "\n";

	//clear training data
	for(int i = 0 ; i < state.num_train_data; i++) {	
		delete [] x[i];
	}

	delete [] x;
	delete [] y;

	return 0;
}

void write_out_model(float** x, int* y, float* alpha, float b) {
	//open output filestream for writing the model
	ofstream model_file;
	model_file.open(state.model_file_name);

	if(model_file.is_open()) {
		//gamma used in kernel for training
		model_file << state.gamma << "\n";

		for(int i=0; i<state.num_train_data; i++) {
			if(alpha[i] != 0) {
				model_file << alpha[i] << "," << y[i];

				for(int j=0; j<state.num_attributes; j++) {
					model_file << "," << x[i][j];
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

float get_train_accuracy(float** x, int* y, float* alpha, float b) {
	int num_correct = 0;

	for(int i=0; i<state.num_train_data; i++) {
		
		//cout << "Iter: " << i << "\n";	
		float dual = 0;

		for(int j=0; j<state.num_train_data; j++) {
			if(alpha[j] != 0) {
				dual += y[j]*alpha[j]*rbf_kernel(x[j], x[i]);
			}
		}

		//dual += b;

		int result = 1;
		if(dual < 0) {
			result = -1;
		}

		if(result == y[i]) {
			num_correct++;
		}
	}

	return ((float)num_correct/(state.num_train_data));
}

float get_duality_gap(float* alpha, int* y, float* f, float c, float b, int num_train_data, int num_attributes) {
	float duality_gap = 0;
	int yi;
	float fi;
	float alpha_i;

	for(int i=0; i<num_train_data; i++) {
		float epsilon;
		yi = y[i];
		fi = f[i];
		alpha_i = alpha[i];

		if(yi == 1) {
			float prod = (0 > (b-fi))?0:(b-fi);
			epsilon = c*prod;
		} else {
			float prod = (0 > (fi-b))?0:(fi-b);
			epsilon = c*prod;
		}

		duality_gap += (alpha_i*yi*fi) + epsilon;
	}

	return duality_gap;
}

void update_f(float* f, float** x, int I_low, int I_hi, int y_low, int y_hi, float alpha_low_old, float alpha_hi_old, float alpha_low_new, float alpha_hi_new) {

	for(int i=0; i<state.num_train_data; i++) {

		float delta = (((alpha_hi_new - alpha_hi_old)*y_hi*rbf_kernel(x[I_hi],x[i])) + ((alpha_low_new - alpha_low_old)*y_low*rbf_kernel(x[I_low],x[i])));

		f[i] += delta;
	}
}

float clip_value(float num, float low, float high) {
	if(num < low) {
		return low;
	} else if(num > high) {
		return high;
	}

	return num;
}

float rbf_kernel(float* x1,float* x2) {
	float* x1_copy = new float[state.num_attributes];

	//deep copy
	get_x(x1, x1_copy, 0, state.num_attributes);
	//get_x(x2, x2_copy, 0, state.num_attributes);

	//TODO: See if BLAS has nicer functions
	cblas_saxpy(state.num_attributes, -1, x2, 1, x1_copy, 1); // x1_copy = -x2_copy + x1_copy

	float norm = cblas_snrm2(state.num_attributes, x1_copy, 1);

	float result = (float)exp(-1 *(double)state.gamma*norm*norm);

	delete [] x1_copy;

	return result;
}

void get_x(float* x, float* x_copy, int idx, int num_attributes) {
	int ctr = 0;

	int start_index = (idx*num_attributes);
	int end_index = start_index+num_attributes;

	for(int i = start_index; i < end_index; i++) {
		x_copy[ctr++] = x[i];
	}
}

void populate_data(float** x, int* y)
{
    ifstream file(state.input_file_name);

    if(!file.is_open())
    {
        cout << "Couldn't open file";
        return;
    }
    //std::vector<std::string>   result;
    string line;
    int curr_example_num = 0;

    while (curr_example_num < state.num_train_data)
    {
        getline(file,line);

        stringstream lineStream(line);
        string cell;

        getline(lineStream,cell,',');

        y[curr_example_num] = stoi(cell);

        int curr_attr_num = 0;

        while(getline(lineStream,cell,','))
        {
            x[curr_example_num][curr_attr_num++] = stof(cell);
        }

        ++curr_example_num;
    }
}

void initialize_f_array(float* f, int* y) {
	for(int i=0; i<state.num_train_data; i++) {
		f[i] = -1*y[i];
	}
}

void set_I_arrays(float* alpha, int* y, vector<int> I[5]) {
	//clear vectors before populating
	for(int i=0; i<5; ++i) {
		I[i].clear();
	}
		
	//populate the I sets
	for(int i=0; i<state.num_train_data; ++i) {
		if(alpha[i] == 0) {
			if(y[i] == 1) {
				I[1].push_back(i);
			} else {
				I[4].push_back(i);
			}
		} else if(alpha[i] == state.c) {
			if(y[i] == -1) {
				I[2].push_back(i);
			} else {
				I[3].push_back(i);
			}
		} else {
			I[0].push_back(i);
		}
	}
}

int get_I_up(float* f, vector<int> I[5]) {
	int I_up = 0;
	float min = 1000000000;

	for(vector<int>::iterator it = I[0].begin(); it != I[0].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}

	for(vector<int>::iterator it = I[1].begin(); it != I[1].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}

	for(vector<int>::iterator it = I[2].begin(); it != I[2].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}
		
	//cout << "Min: " << min << "\n";	
	return I_up;
}

int get_I_low(float* f, vector<int> I[5]) {
	int I_low = 0;
    float max = -1000000000;

    for(vector<int>::iterator it = I[0].begin(); it != I[0].end(); ++it) {
		if(f[*it] > max) {
			max = f[*it];
			I_low = *it;
		}
	}

	for(vector<int>::iterator it = I[3].begin(); it != I[3].end(); ++it) {
		if(f[*it] > max) {
			max = f[*it];
			I_low = *it;
		}
	}

	for(vector<int>::iterator it = I[4].begin(); it != I[4].end(); ++it) {
		if(f[*it] > max) {

//			cout << "\nNew max: " << *it << ": " << f[*it] << "\n";	
			max = f[*it];
			I_low = *it;
		}
	}

	//cout << "Max: " << max << "\n";	
	return I_low;
}

//////////////////////////////// HELPER FUNCTIONS /////////////////////////////////

void print_x(float* x) {

	for(int i = 0; i < state.num_attributes; i++) {
		cout << x[i] << ",";
	}

	cout << "\n";
}
