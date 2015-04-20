#include <stdio.h>
#include <stdlib.h>
#include "svmTrain.h"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cblas.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <string>

#include <thrust/host_vector.h> 
#include <thrust/device_vector.h> 
#include <thrust/copy.h> 
#include <thrust/fill.h> 
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/inner_product.h>
#include <thrust/extrema.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

using namespace std;

void SvmTrain::setup() {

	cout << "A late goodbye";
}

float clip_value(float num, float low, float high);
void populate_data(thrust::host_vector<float> x, thrust::host_vector<int> y);
float rbf_kernel(thrust::host_vector<float> x, int i1, int i2);
void get_x(float* x, float* x_copy, int idx, int num_attributes);

typedef struct {

	int num_attributes;
	int num_train_data;
	float c;
	float gamma;
	float epsilon;
	char input_file_name[30];
	char model_file_name[30];
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

// Scalars
const float alpha = 1;
const float beta = 0;


struct arbitrary_functor
{

	const float C; 

	arbitrary_functor(float _c) : C(_c) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
        // I_set[i] = Alpha[i],  Y[i] , f[i], I_set1[i], I_set2[i];
		if(thrust::get<0>(t) == 0) {
		
			if(thrust::get<1>(t) == 1) {
			
				thrust::get<3>(t) = thrust::get<2>(t);
				
			}
			
			else {
				
				thrust::get<4>(t) = thrust::get<2>(t);
				
			}

		}	else if(thrust::get<0>(t) == C) {
		
			if(thrust::get<1>(t) == -1) {
			
				thrust::get<3>(t) = thrust::get<2>(t);
				
			}
			
			else {
				
				thrust::get<4>(t) = thrust::get<2>(t);
				
			}

		}	else {
		
			thrust::get<3>(t) = thrust::get<2>(t);
			thrust::get<4>(t) = thrust::get<2>(t);
			
		}
	}
};


struct update_functor
{
	const float gamma;
	const float alpha_lo_old;
	const float alpha_hi_old;
	const float alpha_lo_new;
	const float alpha_hi_new;
	const int y_lo;
	const int y_hi;
	const float x_hi_sq;
	const float x_lo_sq;

	update_functor(float _gamma, float _alpha_lo_old, float _alpha_hi_old, float _alpha_lo_new, float _alpha_hi_new, int _y_lo, int _y_hi, float _x_hi_sq, float _x_lo_sq) : 

	gamma(_gamma), 
	alpha_lo_old(_alpha_lo_old), 
	alpha_hi_old(_alpha_hi_new), 
	alpha_lo_new(_alpha_lo_new), 
	alpha_hi_new(_alpha_hi_new), 
	y_lo(_y_lo), 
	y_hi(_y_hi),
	x_hi_sq(_x_hi_sq),
	x_lo_sq(_x_lo_sq) 

	{}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {
		float rbf_hi = expf(-1 * gamma * (thrust::get<2>(t) + x_hi_sq - (2*thrust::get<0>(t)) ));
		float rbf_lo = expf(-1 * gamma * (thrust::get<2>(t) + x_lo_sq - (2*thrust::get<1>(t)) ));

		float delta = (((alpha_hi_new-alpha_hi_old)*y_hi*rbf_hi) + ((alpha_lo_new - alpha_lo_old)*y_lo*rbf_lo));
	
		thrust::get<3>(t) += delta;	
	}
};

int update_f(thrust::device_vector<float> g_f, thrust::device_vector<float> g_x, thrust::device_vector<float> g_x_sq, int I_lo, int I_hi, int y_lo, int y_hi, float alpha_lo_old, float alpha_hi_old, float alpha_lo_new, float alpha_hi_new) {

	cublasStatus_t status;
	cudaError_t cudaStat;
	cublasHandle_t handle;
	
	status = cublasCreate(&handle);
	
	if (status != CUBLAS_STATUS_SUCCESS) { 

		cout << "CUBLAS initialization failed\n"; 
		return EXIT_FAILURE; 
	}

	thrust::device_vector<float> g_hi_dotprod (state.num_train_data);
	thrust::device_vector<float> g_lo_dotprod (state.num_train_data);

	cudaStream_t stream1;
	cudaStream_t stream2;

	cudaStat = cudaStreamCreate(&stream1);
	cudaStat = cudaStreamCreate(&stream2);

	//Allocate x_hi, x_lo and an empty vector in device	i

	float* raw_g_x = thrust::raw_pointer_cast(&g_x[0]);
	float* raw_g_f = thrust::raw_pointer_cast(&g_f[0]);
	float* raw_g_hi_dotprod = thrust::raw_pointer_cast(&g_hi_dotprod[0]);
	float* raw_g_lo_dotprod = thrust::raw_pointer_cast(&g_lo_dotprod[0]);

	status = cublasSetStream(handle, stream1);

	status = cublasSgemv( handle, CUBLAS_OP_T, state.num_train_data, state.num_attributes, &alpha, raw_g_x, state.num_attributes, &raw_g_x[I_hi * state.num_attributes], 1, &beta, raw_g_hi_dotprod, 1 );

	cublasSetStream(handle, stream2);
	
	status = cublasSgemv( handle, CUBLAS_OP_T, state.num_train_data, state.num_attributes, &alpha, raw_g_x, state.num_attributes, &raw_g_x[I_lo * state.num_attributes], 1, &beta, raw_g_lo_dotprod, 1 );

	float x_hi_sq = g_x_sq[I_hi];
	float x_lo_sq = g_x_sq[I_lo];
		
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(g_hi_dotprod.begin(), g_lo_dotprod.begin(), g_x_sq.begin(), g_f.begin())),
   	                 thrust::make_zip_iterator(thrust::make_tuple(g_hi_dotprod.end(), g_lo_dotprod.end(), g_x_sq.end(),g_f.end())),
       	             update_functor(state.gamma, alpha_lo_old, alpha_hi_old, alpha_lo_new, alpha_hi_new, y_lo, y_hi, x_hi_sq, x_lo_sq));


/////////////////////////////////////////////////////////

	cublasDestroy( handle );

	return 0;
}


int main(int argc, char *argv[]) {

	//Obtain the command line arguments
	parse_arguments(argc, argv);

	//input data attributes and labels
	thrust::host_vector<float> x (state.num_train_data * state.num_attributes);
	thrust::host_vector<int> y (state.num_train_data);
	
	//read data from input file
	populate_data(x, y);

	cout << "Populated Data from input file\n";

	//Copy x and y to device
	thrust::device_vector<float> g_x (x.begin(), x.end());
	thrust::device_vector<int> g_y(y.begin(), y.end());
	
	thrust::device_vector<float> g_x_hi(state.num_attributes);
	thrust::device_vector<float> g_x_lo(state.num_attributes);
	
	// Initialize f on device
	thrust::device_vector<float> g_f(state.num_train_data);
	thrust::transform(g_f.begin(), g_f.end(), g_y.begin(), thrust::negate<float>());

	//Initialize alpha on device
	thrust::device_vector<int> g_alpha(state.num_train_data, 0);
	
	//b (intercept), checks optimality condition for stopping
	float b_lo, b_hi;

	//check iteration number for stopping condition
	int num_iter = 0;
 
	thrust::host_vector<float> g_x_sq (state.num_train_data);	

	for( int i = 0; i < state.num_train_data; i++ )
	{
		g_x_sq[i] = thrust::inner_product(&g_x[i*state.num_attributes], &g_x[i*state.num_attributes] + state.num_attributes, &g_x[i*state.num_attributes], 0.0f);
	}
	
	thrust::device_vector<float>::iterator iter;
	
	do {

		//Set up I_set1 and I_set2
		thrust::device_vector<int> g_I_set1(state.num_train_data, 1000000000);
		thrust::device_vector<int> g_I_set2(state.num_train_data, -1000000000);
		
		thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(g_alpha.begin(), g_y.begin(), g_f.begin(), g_I_set1.begin(), g_I_set2.begin())),
    	                 thrust::make_zip_iterator(thrust::make_tuple(g_alpha.end(), g_y.end(), g_f.end(), g_I_set1.end(), g_I_set2.end())),
        	             arbitrary_functor(state.c));

		//get b_hi and b_low
		iter = thrust::max_element(g_I_set2.begin(), g_I_set2.end());

		int I_lo = iter - g_I_set2.begin();
		b_lo = *iter;

		iter = thrust::min_element(g_I_set1.begin(), g_I_set1.end());

		int I_hi = iter - g_I_set1.begin();
		b_hi = *iter;

		int y_lo = y[I_lo];
		int y_hi = y[I_hi];

		float eta = rbf_kernel(x,I_hi,I_hi) + rbf_kernel(x,I_lo,I_lo) - (2*rbf_kernel(x,I_lo,I_hi)) ;

		//obtain alpha_low and alpha_hi (old values)
		float alpha_lo_old = g_alpha[I_lo];
		float alpha_hi_old = g_alpha[I_hi];

		//update alpha_low and alpha_hi
		float s = y_lo*y_hi;
		float alpha_lo_new = alpha_lo_old + (y_lo*(b_hi - b_lo)/eta);
		float alpha_hi_new = alpha_hi_old + (s*(alpha_lo_old - alpha_lo_new));

		//clip new alpha values between 0 and C
		alpha_lo_new = clip_value(alpha_lo_new, 0.0, state.c);
		alpha_hi_new = clip_value(alpha_hi_new, 0.0, state.c);
		
		//store new alpha_1 and alpha_2 values
		g_alpha[I_lo] = alpha_lo_new;
		g_alpha[I_hi] = alpha_hi_new;

		//update f values
		update_f(g_f, g_x, g_x_sq, I_lo, I_hi, y_lo, y_hi, alpha_lo_old, alpha_hi_old, alpha_lo_new, alpha_hi_new);

		//Increment number of iterations to reach stopping condition
		num_iter++;

		cout << "Current iteration number: " << num_iter << "\n";

	} while((b_lo > (b_hi +(2*state.epsilon))) && num_iter < state.max_iter);

	if(b_lo > (b_hi + (2*state.epsilon))) {
		cout << "Could not converge in " << num_iter << " iterations. SVM training has been stopped\n";
	} else {
		cout << "Converged at iteration number: " << num_iter << "\n";
	}

	//obtain final b intercept
	float b = (b_lo + b_hi)/2;
	cout << "b: " << b << "\n";

	//obtain training accuracy
	//float train_accuracy = get_train_accuracy(x, y, alpha, b);
	//cout << "Training accuracy: " << train_accuracy << "\n";

	//write model to file
	//write_out_model(x, y, alpha, b);

	//cout << "Training model has been saved to the file " << state.model_file_name << "\n";

	//clear training data
	//for(int i = 0 ; i < state.num_train_data; i++) {	
	//	delete [] x[i];
	//}

	//delete [] x;
	//delete [] y;

	return 0;
}


float clip_value(float num, float low, float high) {
	if(num < low) {
		return low;
	} else if(num > high) {
		return high;
	}

	return num;
}


void populate_data(thrust::host_vector<float> x, thrust::host_vector<int> y)
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
            x[(curr_example_num * state.num_attributes) + curr_attr_num++] = stof(cell);
        }

        ++curr_example_num;
    }
}


void get_x(float* x, float* x_copy, int idx, int num_attributes) {
	int ctr = 0;

	int start_index = (idx*num_attributes);
	int end_index = start_index+num_attributes;

	for(int i = start_index; i < end_index; i++) {
		x_copy[ctr++] = x[i];
	}
}


float rbf_kernel(thrust::host_vector<float> x, int i1, int i2){
	
	float* i2_copy = new float[state.num_attributes];

	float* raw_i1 = thrust::raw_pointer_cast(&x[i1*state.num_attributes]);
	float* raw_i2 = thrust::raw_pointer_cast(&x[i2*state.num_attributes]);

	get_x(raw_i2, i2_copy, 0, state.num_attributes);
	
	cblas_saxpy(state.num_attributes, -1, raw_i1, 1, i2_copy, 1); 

	float norm_sq = cblas_sdot(state.num_attributes, i2_copy, 1, i2_copy, 1);

	float result = (float)exp(-1 *(double)state.gamma*norm_sq);

	delete [] i2_copy;

	return result;
}
