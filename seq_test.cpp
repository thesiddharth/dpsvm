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

void populate_data(float** x, int* y);
void get_x(float* x, float* x_copy, int idx, int num_attributes);
float rbf_kernel(float* x1,float* x2);
float get_test_accuracy(float** x, int* y, float* alpha, float** x_model, int* y_model, int num_sv);
int get_num_sv();
void populate_model(float** x_model, int* y_model, float* alpha, int num_sv);

typedef struct {

	int num_attributes;
	int num_test_data;
	float gamma;
	char input_file_name[30];
	char model_file_name[30];

} state_model;

//global structure for training parameters
static state_model state;

static void usage_exit() {
    cerr <<
"   Command Line:\n"
"\n"
"   -a/--num-att        :  [REQUIRED] The number of attributes\n"
"									  /features\n"
"   -x/--num-ex       	:  [REQUIRED] The number of testing \n"
"									  examples\n"
"   -f/--file-path      :  [REQUIRED] Path to the testing file\n"
"   -g/--gamma       	:  Parameter gamma of the radial basis\n"
"						   function: exp(-gamma*|u-v|^2)\n"
"						   (default: 1/num-att)"
"	-m/--model 			:  [REQUIRED] Path of model (output of training phase)\n"
"\n";
    
	exit(-1);
}

static struct option longOptionsG[] =
{
    { "num-att",        required_argument,          0,  'a' },
    { "num-ex",         required_argument,          0,  'x' },
    { "gamma",          required_argument,          0,  'g' },
    { "file-path",      required_argument,          0,  'f' },
    { "model",			required_argument,			0,	'm' },
    { 0,                0,                          0,   0  }
};

static void parse_arguments(int argc, char* argv[]) {

    // Default Values
	state.num_attributes = -1;
	state.num_test_data = -1;
	state.gamma = -1;
	strcpy(state.input_file_name, "");
	strcpy(state.model_file_name, "");

    // Parse args
    while (1) {
        int idx = 0;
        int c = getopt_long(argc, argv, "a:x:g:f:m:", longOptionsG, &idx);

        if (c == -1) {
            // End of options
            break;
        }

        switch (c) {
        case 'a':
            state.num_attributes = atoi(optarg);
            break;
        case 'x':
            state.num_test_data = atoi(optarg);
            break;
        case 'g':
            state.gamma = atof(optarg);
            break;
       case 'f':
            strcpy(state.input_file_name, optarg);
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

	if(state.num_attributes <= 0 || state.num_test_data <= 0) {

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
	float** x = new float*[state.num_test_data];
	for(int i = 0 ; i < state.num_test_data; i++) {
		x[i] = new float[state.num_attributes]();
	}

	int* y = new int[state.num_test_data];

	//read data from input file
	populate_data(x, y);

	cout << "Populated test data\n";

	//get no. of support vectors
	int num_sv = get_num_sv();

	cout << "Total number of Support Vectors: " << num_sv << "\n";

	//training model data
	//input data attributes and labels
	float** x_model = new float*[num_sv];
	for(int i = 0 ; i < num_sv; i++) {
		x_model[i] = new float[state.num_attributes]();
	}
	int* y_model = new int[num_sv];
	float* alpha = new float[num_sv];

	//read data from model file
	populate_model(x_model, y_model, alpha, num_sv);

	cout << "Populated training model\n";

	//obtain testing accuracy
	float test_accuracy = get_test_accuracy(x, y, alpha, x_model, y_model, num_sv);
	cout << "Test accuracy: " << test_accuracy << "\n";

	//clear test data
	for(int i = 0 ; i < state.num_test_data; i++) {	
		delete [] x[i];
	}

	delete [] x;
	delete [] y;

	//clear model data
	for(int i = 0 ; i < num_sv; i++) {	
		delete [] x_model[i];
	}

	delete [] x_model;
	delete [] y_model;
	delete [] alpha;

	return 0;
}

float get_test_accuracy(float** x, int* y, float* alpha, float** x_model, int* y_model, int num_sv) {
	int num_correct = 0;

	for(int i=0; i<state.num_test_data; i++) {
		float dual = 0;

		for(int j=0; j<num_sv; j++) {
			dual += y_model[j]*alpha[j]*rbf_kernel(x_model[j], x[i]);
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

	return ((float)num_correct/(state.num_test_data));
}

void populate_model(float** x_model, int* y_model, float* alpha, int num_sv) {
	ifstream file(state.model_file_name);

    if(!file.is_open())
    {
        cout << "Couldn't open model file";
        exit(-1);
    }
    //std::vector<std::string>   result;
    string line;
    int curr_example_num = 0;

    //get gamma
    getline(file, line);
    state.gamma = stof(line);

    while (curr_example_num < num_sv)
    {
        getline(file,line);

        stringstream lineStream(line);
        string cell;

        getline(lineStream,cell,',');
        alpha[curr_example_num] = stof(cell);

        getline(lineStream,cell,',');
        y_model[curr_example_num] = stoi(cell);

        int curr_attr_num = 0;

        while(getline(lineStream,cell,',')) {
            x_model[curr_example_num][curr_attr_num++] = stof(cell);
        }

        ++curr_example_num;
    }
}

int get_num_sv() {
	int num_sv = 0;

	string sv_line;
    ifstream model_file(state.model_file_name);

    if(!model_file.is_open()) {
    	cout << "Model file " << state.model_file_name << " couldn't be opened.\n";
    	exit(-1);
    }

    while (getline(model_file, sv_line)) {
    	num_sv++;
    }

    //decrement a count for gamma
    num_sv --;

    return num_sv;
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

    while (curr_example_num < state.num_test_data)
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