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
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include "CycleTimer.h"
#include "cache.hpp"
#include "svmTest.h"

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/copy.h> 
#include <thrust/fill.h> 
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

// Scalars
const float alpha_const = 1;
const float beta_const = 0;

void populate_data(vector<float> &x, vector<int> &y);
void get_x(float* x, float* x_copy, int idx, int num_attributes);
float rbf_kernel(float* x1,float* x2);
float get_test_accuracy(float** x, int* y, float* alpha, float** x_model, int* y_model, float b, int num_sv);
int get_num_sv();
void populate_model(vector<float> &x_model, vector<int> &y_model, vector<float> &alpha, float &b, int num_sv);

typedef struct {

	int num_attributes;
	int num_test_data;
	float gamma;
	char input_file_name[60];
	char model_file_name[60];

} state_model;

//global structure for training parameters
static state_model state;

static void usage_exit() {
    cerr <<
"   Command Line:\n"
"\n"
"   -a/--num-att        :  [REQUIRED] The number of attributes\n"
"                                     /features\n"
"   -x/--num-ex         :  [REQUIRED] The number of testing \n"
"                                     examples\n"
"   -f/--file-path      :  [REQUIRED] Path to the testing file\n"
"   -g/--gamma          :  Parameter gamma of the radial basis\n"
"                          function: exp(-gamma*|u-v|^2)\n"
"                          (default: 1/num-att)\n"
"   -m/--model          :  [REQUIRED] Path of model (output of training phase)\n"
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
    vector<float> raw_x(state.num_test_data*state.num_attributes, 0.0f);
    vector<int> raw_y(state.num_test_data, 0);

	float b = 0;

	//read data from input file
	populate_data(raw_x, raw_y);

	cout << "Populated test data\n";

	//get no. of support vectors
	int num_sv = get_num_sv();

	cout << "Total number of Support Vectors: " << num_sv << "\n";

	//training model data
	//input data attributes and labels
    vector<float> raw_x_model(num_sv*state.num_attributes, 0.0f);
    vector<int> raw_y_model(num_sv, 0);
    vector<float> raw_alpha(num_sv, 0);

	//read data from model file
	populate_model(raw_x_model, raw_y_model, raw_alpha, b, num_sv);

	cout << "Populated training model\n";

    //instantiate a class of SvmTest
    SvmTest svmTest(b, num_sv, state.num_test_data, state.num_attributes, state.gamma);

    //set up the svmTest object with test and model data
    svmTest.setup(raw_x, raw_y, raw_x_model, raw_y_model, raw_alpha);

    cout << "CUDA setup complete\n";

	//obtain testing accuracy
	float test_accuracy = svmTest.get_test_accuracy();
	cout << "Test accuracy: " << test_accuracy << "\n";

    //clear handles
    svmTest.destroy_cuda_handles();

	//clear test and model data
    raw_x.clear();
    raw_y.clear();

    raw_x.shrink_to_fit();
    raw_y.shrink_to_fit();

    raw_x_model.clear();
    raw_y_model.clear();
    raw_alpha.clear();

    raw_x_model.shrink_to_fit();
    raw_y_model.shrink_to_fit();
    raw_alpha.shrink_to_fit();

	return 0;
}


void populate_model(vector<float> &x_model, vector<int> &y_model, vector<float> &alpha, float &b, int num_sv) {
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
    state.gamma = std::stof(line);

    //get b
    getline(file, line);
    b = std::stof(line);

    while (curr_example_num < num_sv)
    {
        getline(file,line);

        stringstream lineStream(line);
        string cell;

        getline(lineStream,cell,',');
        alpha[curr_example_num] = std::stof(cell);

        getline(lineStream,cell,',');
        y_model[curr_example_num] = std::stoi(cell);

        //start index of x_model
        int idx = curr_example_num * state.num_attributes;

        while(getline(lineStream,cell,',')) {
            x_model[idx++] = std::stof(cell);
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

    //decrement counts for gamma and b
    num_sv = num_sv - 2;

    return num_sv;
}

void populate_data(vector<float> &x, vector<int> &y)
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

        y[curr_example_num] = std::stoi(cell);

        //start index of x
        int idx = curr_example_num * state.num_attributes;
        
        while(getline(lineStream,cell,','))
        {
            x[idx++] = std::stof(cell);
        }

        ++curr_example_num;
    }
}

//Parameterized constructor
SvmTest::SvmTest(float model_b, int model_num_sv, int num_test_data_ip, int model_num_attributes, float model_gamma) {
    b = model_b;
    num_sv = model_num_sv;
    num_test_data = num_test_data_ip;
    num_attributes = model_num_attributes;
    gamma = model_gamma;
}

void SvmTest::setup(std::vector<float>& raw_x, std::vector<int>& raw_y, std::vector<float>& raw_x_model, std::vector<int>& raw_y_model, std::vector<float>& raw_alpha) {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for DPSVM Test\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    //test data transfer to device
    x = thrust::host_vector<float>(raw_x);
    y = thrust::host_vector<int>(raw_y);

    g_x = thrust::device_vector<float>(x.begin(), x.end());
    g_y = thrust::device_vector<int>(y.begin(), y.end());

    //model data transfer to device
    x_model = thrust::host_vector<float>(raw_x_model);
    y_model = thrust::host_vector<int>(raw_y_model);
    alpha = thrust::host_vector<float>(raw_alpha);

    g_x_model = thrust::device_vector<float>(x_model.begin(), x_model.end());
    g_y_model = thrust::device_vector<int>(y_model.begin(), y_model.end());
    g_alpha = thrust::device_vector<float>(alpha.begin(), alpha.end());

    init_cuda_handles();

    //calculate dot products (2 norm) of x in test and model
    g_x_sq = thrust::device_vector<float>(num_test_data);
    for(int i=0; i<num_test_data; i++) {
        g_x_sq[i] = thrust::inner_product(&g_x[i*num_attributes], &g_x[i*num_attributes] + num_attributes, &g_x[i*num_attributes], 0.0f);
    }

    g_x_sq_model = thrust::device_vector<float>(num_sv);
    for(int i=0; i<num_sv; i++) {
        g_x_sq_model[i] = thrust::inner_product(&g_x_model[i*num_attributes], &g_x_model[i*num_attributes] + num_attributes, &g_x_model[i*num_attributes], 0.0f);
    }
}

template <typename Tuple>
struct test_functor : public thrust::unary_function<float,Tuple> {

    const float i_sq;
    const float gamma;

    test_functor(float _i_sq, float _gamma) : 

    i_sq(_i_sq),
    gamma(_gamma) 

    {}


    __host__ __device__ float operator()(const Tuple& t) const
    {
      return (thrust::get<0>(t) * thrust::get<1>(t) * expf(-1 * gamma * (thrust::get<2>(t) + i_sq - (2*thrust::get<3>(t)))));
    }
};

float SvmTest::get_test_accuracy() {
    int num_correct = 0;

    raw_g_x = thrust::raw_pointer_cast(&g_x[0]);
    raw_g_x_c = thrust::raw_pointer_cast(&g_x_model[0]);

    g_t_dp = thrust::device_vector<float>(num_sv);
    raw_g_t_dp = thrust::raw_pointer_cast(&g_t_dp[0]);
    
    for(int i=0; i<num_test_data; i++) {

    
        cublasSgemv(handle, CUBLAS_OP_T, num_attributes, num_sv, &alpha_const, &raw_g_x_c[0], num_attributes, &raw_g_x[i * num_attributes], 1, &beta_const, raw_g_t_dp, 1);


        float i_sq = g_x_sq[i];

    
        float dual = 0;

        dual = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(g_y_model.begin(), g_alpha.begin(), g_x_sq_model.begin(), g_t_dp.begin())),
                     thrust::make_zip_iterator(thrust::make_tuple(g_y_model.end(), g_alpha.end(), g_x_sq_model.end(), g_t_dp.end())),
                     test_functor<thrust::tuple<int, float, float, float> >(i_sq, gamma), 0.0f, thrust::plus<float>());

        dual -= b;

        int result = 1;
        if(dual < 0) {
            result = -1;
        }

        if(result == y[i]) {
            num_correct++;
        }
    }

    return ((float)num_correct/(num_test_data));
}

void SvmTest::init_cuda_handles() {

    cublasStatus_t status;
    cudaError_t cudaStat;
    
    status = cublasCreate(&handle);
    
    if (status != CUBLAS_STATUS_SUCCESS) { 

        cout << "CUBLAS initialization failed\n"; 
        exit(EXIT_FAILURE); 
    }

    cudaStat = cudaStreamCreate(&stream1);
    cudaStat = cudaStreamCreate(&stream2);

    if (cudaStat == cudaErrorInvalidValue) { 

        cout << "CUDA stream initialization failed\n"; 
        exit(EXIT_FAILURE); 
    }
    
}

void SvmTest::destroy_cuda_handles() {

    cublasDestroy(handle);

}
