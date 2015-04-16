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

using namespace std;

void populate_data(char* input_file_name, float* x, int* y, int num_attributes, int num_train_data);
void initialize_f_array(float* f, int* y, int num_train_data);
void set_I_arrays(float* alpha, int *y, float c, int num_train_data, vector<int> I[5]);
int get_I_up(float* f, vector<int> I[5]);
int get_I_low(float* f, vector<int> I[5]);
void get_x(float* x, float* x_copy, int idx, int num_attributes);
float rbf_kernel(float* x1,float* x2);
float clip_value(float num, float low, float high);
void update_f(float* f, float* x, float* x_low, float* x_hi, int y_low, int y_hi, float alpha_low_old, float alpha_hi_old, float alpha_low_new, float alpha_hi_new);
float get_duality_gap(float* alpha, int* y, float* f, float c, float b, int num_train_data, int num_attributes);
void print_x(float* x);
float get_train_accuracy(float* x, int* y, float* alpha, float b);

typedef struct {

	int num_attributes;
	int num_train_data;
	float c;
	float gamma;
	float epsilon;
	char input_file_name[30];

} state_model;

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

    // Parse args
    while (1) {
        int idx = 0;
        int c = getopt_long(argc, argv, "a:x:c:g:f:e:", longOptionsG, &idx);

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
        default:
            cerr << "\nERROR: Unknown option: -" << c << "\n";
            // Usage exit
            usage_exit();
        }
    }

	if(strcmp(state.input_file_name,"")==0) {

		cerr << "Enter a valid file name\n";
		usage_exit();
	}

	if(state.num_attributes <= 0 || state.num_train_data <= 0) {

		cerr << "Missing a required parameter, or invalid parameter\n";
		usage_exit();

	}

	if(state.gamma < 0) {

		state.gamma = 1 / state.num_train_data;
	}

}

int main(int argc, char *argv[]) {

	parse_arguments(argc, argv);
	
	//TODO: command line arguments here
	int num_attributes = state.num_attributes;
	int num_train_data = state.num_train_data;
	float C = state.c;
	char input_file_name[30];
	strcpy(input_file_name, state.input_file_name);
	float tolerance = state.epsilon;

	//input data attributes and labels
	float* x = new float[num_attributes*num_train_data];
	int* y = new int[num_train_data];

	//read data from input file
	populate_data(input_file_name, x, y, num_attributes, num_train_data);

	cout << "Populated Data" << "\n";

	//Initialize starting values
	float* alpha = new float[num_train_data]();
	float* f = new float[num_train_data];

	initialize_f_array(f, y, num_train_data);

	//I sets
	vector<int> I[5];

	//b (intercept)
	float b_low, b_hi;

	int num_iter = 0;

	float eta, prev_eta;
	eta = 0;

	do {
		prev_eta = eta;

		//update the I sets
		set_I_arrays(alpha, y, C, num_train_data, I);
	
		cout << I[0].size() << "\n";
		cout << I[1].size() << "\n";
		cout << I[2].size() << "\n";
		cout << I[3].size() << "\n";
		cout << I[4].size() << "\n";


		//get alpha1 and alpha2
		int I_hi = get_I_up(f,I);
		b_hi = f[I_hi];

		int I_low = get_I_low(f,I);
		b_low = f[I_low];

		cout << "------------------------\n";
		cout << "I_hi: " << I_hi << "\n";
		cout << "I_low: " << I_low << "\n";
		cout << "------------------------\n";

		float* x_low = new float[num_attributes];
		float* x_hi = new float[num_attributes];

		get_x(x, x_low, I_low, num_attributes);
		get_x(x, x_hi, I_hi, num_attributes);

		int y_low = y[I_low];
		int y_hi = y[I_hi];

		eta = (2*rbf_kernel(x_low,x_hi)); //- rbf_kernel(x_hi,x_hi) + rbf_kernel(x_low,x_low) - ;
		
		cout << "------------------------\n";
		cout << "Eta: " << eta << "\n";
		cout << "------------------------\n";

		float alpha_low_old = alpha[I_low];
		float alpha_hi_old = alpha[I_hi];

		//update alpha1 and alpha2
		float s = y_low*y_hi;
		float alpha_low_new = alpha_low_old + (y_low*(b_hi - b_low)/eta);
		float alpha_hi_new = alpha_hi_old + (s*(alpha_low_old - alpha_low_new));

		cout << "------------------------\n";
		cout << "Pre clip aplha low: " << alpha_low_new << "\n";
		cout << "Pre clip aplha hi: " << alpha_hi_new << "\n";	
		cout << "------------------------\n";

		//clip alpha values between 0 and C
		alpha_low_new = clip_value(alpha_low_new, 0.0, C);
		alpha_hi_new = clip_value(alpha_hi_new, 0.0, C);

		cout << "------------------------\n";
		cout << "Post clip aplha low: " << alpha_low_new << "\n";
		cout << "Post clip aplha hi: " << alpha_hi_new << "\n";	
		cout << "------------------------\n";
		
		//store new alpha_1 and alpha_2 values
		alpha[I_low] = alpha_low_new;
		alpha[I_hi] = alpha_hi_new;

		//update f values
		update_f(f, x, x_low, x_hi, y_low, y_hi, alpha_low_old, alpha_hi_old, alpha_low_new, alpha_hi_new);

		//obtain new dual
		/*float dual_old = dual;
		dual = dual_old - (((alpha1_new - alpha1_old)/y1)*(b_low - b_up)) + ((eta/2)*((alpha1_new - alpha1_old)/y1) * ((alpha1_new - alpha1_old)/y1));

		cout << "------------------------\n";
		cout << "Dual: " << dual << "\n";	
		cout << "------------------------\n";

		//get b
		b = (b_up + b_low) / 2;

		cout << "------------------------\n";
		cout << "b: " << b << "\n";
		cout << "------------------------\n";

		//obtain the new duality gap
		duality_gap = get_duality_gap(alpha, y, f, C, b, num_train_data, num_attributes);

		cout << "Duality gap: " << duality_gap << "\n";*/

		delete [] x_low;
		delete [] x_hi;

		cout << "------------------------\n";
		cout << "b_low: " << b_low << "\n";
		cout << "b_hi + 2* tol: " << b_hi+(2*tolerance) << "\n";
		cout << "------------------------\n";

		num_iter++;

		cout << "Current iteration number: " << num_iter << "\n";

	} while((b_low > (b_hi +(2*tolerance))) && (prev_eta != eta));

	cout << "Converged at iteration number: " << num_iter << "\n";

	float b = (b_low + b_hi)/2;

	float train_accuracy = get_train_accuracy(x, y, alpha, b);

	cout << "Training accuracy: " << train_accuracy << "\n";

	return 0;
}

float get_train_accuracy(float* x, int* y, float* alpha, float b) {
	int num_correct = 0;

	for(int i=0; i<state.num_train_data; i++) {
		float dual = 0;
		
		float* x_i = new float[state.num_attributes];
		get_x(x, x_i, i, state.num_attributes);

		for(int j=0; j<state.num_train_data; j++) {
			float* x_j = new float[state.num_attributes];
			get_x(x, x_j, j, state.num_attributes);

			dual += y[j]*alpha[j]*rbf_kernel(x_j, x_i);

			delete [] x_j;
		}

		dual += b;

		int result = 1;
		if(dual < 0) {
			result = -1;
		}

		if(result == y[i]) {
			num_correct++;
		}

		delete [] x_i;
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

void update_f(float* f, float* x, float* x_low, float* x_hi, int y_low, int y_hi, float alpha_low_old, float alpha_hi_old, float alpha_low_new, float alpha_hi_new) {
	
	float* xi = new float[state.num_attributes];
	
	for(int i=0; i<state.num_train_data; i++) {
		get_x(x, xi, i, state.num_attributes);

		float delta = (((alpha_hi_new - alpha_hi_old)*y_hi*rbf_kernel(x_hi,xi)) + ((alpha_low_new - alpha_low_old)*y_low*rbf_kernel(x_low,xi)));

		f[i] += delta;
	}

	delete [] xi;
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
	float* x2_copy = new float[state.num_attributes];

	//deep copy
	get_x(x1, x1_copy, 0, state.num_attributes);
	get_x(x2, x2_copy, 0, state.num_attributes);

	//TODO: See if BLAS has nicer functions
	cblas_saxpy(state.num_attributes, -1, x2_copy, 1, x1_copy, 1); // x1_copy = -x2_copy + x1_copy

	float norm = cblas_snrm2(state.num_attributes, x1_copy, 1);

	//float result = (float)exp((double)state.gamma*norm*norm);

	float result = (float)exp(-1 *(double)state.gamma*norm*norm);

	//cout << "Norm: " << norm << "\n";
	//cout << "Res1: " << result <<"\n";
	//cout << "Res2: " << result2 << "\n";

	//exit(0);

	delete [] x1_copy;
	delete [] x2_copy;

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

void populate_data(char* input_file_name, float* x, int* y, int num_attributes, int num_train_data)
{
    ifstream file(input_file_name);

    if(!file.is_open())
    {
        cout << "Couldn't open file";
        return;
    }
    //std::vector<std::string>   result;
    string line;
    int curr_example_num = 0;

    while (curr_example_num < num_train_data)
    {
        getline(file,line);

        stringstream lineStream(line);
        string cell;

        getline(lineStream,cell,',');

        y[curr_example_num] = stoi(cell);

        int curr_attr_num = 0;

        while(getline(lineStream,cell,','))
        {
            x[(curr_example_num * num_attributes) + curr_attr_num++] = stof(cell);
        }

        ++curr_example_num;
    }
}

void initialize_f_array(float* f, int* y, int num_train_data) {
	for(int i=0; i<num_train_data; i++) {
		f[i] = -1*y[i];
	}
}

void set_I_arrays(float* alpha, int* y, float c, int num_train_data, vector<int> I[5]) {
	//clear vectors before populating
	for(int i=0; i<5; ++i) {
		I[i].clear();
	}
		
	//populate the I sets
	for(int i=0; i<num_train_data; ++i) {
		if(alpha[i] == 0) {
			if(y[i] == 1) {
				I[1].push_back(i);
			} else {
				I[4].push_back(i);
			}
		} else if(alpha[i] == c) {
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
			max = f[*it];
			I_low = *it;
		}
	}
	
	return I_low;
}

//////////////////////////////// HELPER FUNCTIONS /////////////////////////////////

void print_x(float* x) {

	for(int i = 0; i < state.num_attributes; i++) {
		cout << x[i] << ",";
	}

	cout << "\n";
}
