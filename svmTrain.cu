#include <stdio.h>
#include <stdlib.h>
#include "svmTrain.h"
#include "parse.hpp"
#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cblas.h>
#include <vector>
#include <string.h>
#include <getopt.h>
#include <math.h>
#include <vector>
#include "CycleTimer.h"
#include "svmTrainMain.hpp"
#include "cache.hpp"

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
const float alpha = 1;
const float beta = 0;

//functor for obtaining the I sets
struct arbitrary_functor
{

	const float C; 

	arbitrary_functor(float _c) : C(_c) {}

    template <typename Tuple>
    __host__ __device__
    void operator()(Tuple t)
    {

		thrust::get<3>(t).I_1 = thrust::get<3>(t).I_2 = thrust::get<4>(t);
		//i_helper new;
        // I_set[i] = Alpha[i],  Y[i] , f[i], I_set1[i], I_set2[i];
		if(thrust::get<0>(t) == 0) {
		
			if(thrust::get<1>(t) == 1) {
			
				thrust::get<3>(t).f_1 = thrust::get<2>(t); 
				thrust::get<3>(t).f_2 = -1000000000; 	
			}
			
			else {
				
				thrust::get<3>(t).f_2 = thrust::get<2>(t); 
				thrust::get<3>(t).f_1 = 1000000000;
					
			}

		}	else if(thrust::get<0>(t) == C) {
		
			if(thrust::get<1>(t) == -1) {
			
				thrust::get<3>(t).f_1 = thrust::get<2>(t); 
				thrust::get<3>(t).f_2 = -1000000000; 	
				
			}
			
			else {
				
				thrust::get<3>(t).f_2 = thrust::get<2>(t); 
				thrust::get<3>(t).f_1 = 1000000000;
				
			}

		}	else {
		
				thrust::get<3>(t).f_1 = thrust::get<3>(t).f_2 = thrust::get<2>(t); 
			
		}

		
	}
};

//functor for performing the f_update step in GPU using Thrust
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
	alpha_hi_old(_alpha_hi_old), 
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
		//printf("%f\t%f\n" , -1 * gamma * (thrust::get<2>(t) + x_hi_sq - (2*thrust::get<0>(t)) ) , rbf_hi);
		float rbf_lo = expf(-1 * gamma * (thrust::get<2>(t) + x_lo_sq - (2*thrust::get<1>(t)) ));
		//printf("%f\t%f\n" , -1 * gamma * (thrust::get<2>(t) + x_lo_sq - (2*thrust::get<1>(t)) ) , rbf_lo);

		float delta = (((alpha_hi_new-alpha_hi_old)*y_hi*rbf_hi) + ((alpha_lo_new - alpha_lo_old)*y_lo*rbf_lo));
	
		thrust::get<3>(t) += delta;	
	}
};



//cache lookup
thrust::device_vector<float>& SvmTrain::lookup_cache(int I_idx, bool& cache_hit) {

	//static thrust::device_vector<float> g_hi_dotprod (state.num_train_data);
	thrust::device_vector<float>* lookup = lineCache->lookup(I_idx);
	if(lookup != NULL){
		cache_hit = true;
		return *lookup;
	}

	else {
		cache_hit = false;
		return lineCache->get_new_cache_line(I_idx);

	}
}

//Allocate x_hi, x_lo and an empty vector in device	i
void SvmTrain::init_cuda_handles() {

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

void SvmTrain::destroy_cuda_handles() {

	cublasDestroy(handle);

}


int SvmTrain::update_f(int I_lo, int I_hi, int y_lo, int y_hi, float alpha_lo_old, float alpha_hi_old, float alpha_lo_new, float alpha_hi_new) {

//	unsigned long long t1,t2;
//	t1 = CycleTimer::currentTicks();
	
	//	cout << I_hi << "," << I_lo << "\n";

	//	lineCache -> dump_map_contents();	


	bool hi_hit;
	bool lo_hit;

	thrust::device_vector<float>& g_hi_dotprod  = lookup_cache(I_hi, hi_hit);
	
	float* raw_g_hi_dotprod = thrust::raw_pointer_cast(&g_hi_dotprod[0]);

	//printf("%x, %x\n",raw_g_hi_dotprod, raw_g_lo_dotprod);

	//cout << "UPDATE_F: " << t2-t1 << "\n";
	//t1 = t2;

	if(!hi_hit) {

		//cout << "HI MISS\n";

		cublasSetStream(handle, stream1);

//	t2 = CycleTimer::currentTicks();
//	cout << "UPDATE_F, INIT: " << t2-t1 << "\n";
//	t1 = t2;
		
		cublasSgemv( handle, CUBLAS_OP_T, state.num_attributes, num_train_data, &alpha, &raw_g_x[matrix_start], state.num_attributes, &raw_g_x[I_hi * state.num_attributes], 1, &beta, raw_g_hi_dotprod, 1 );
	
//	t2 = CycleTimer::currentTicks();
//	cout << "SGEMV 1: " << t2-t1 << "\n";
//	t1 = t2;
	}

	/*cout << "----------------\n";

	for (int i = 100 ; i < 130; i++) {

		cout << g_hi_dotprod[i] << ",";

	}

	cout << "\n-------------\n";*/
	thrust::device_vector<float>& g_lo_dotprod  = lookup_cache(I_lo, lo_hit);
	float* raw_g_lo_dotprod = thrust::raw_pointer_cast(&g_lo_dotprod[0]);
	
	if(!lo_hit) {

		//cout << "LO MISS \n";

		cublasSetStream(handle, stream2);
	
		cublasSgemv( handle, CUBLAS_OP_T, state.num_attributes, num_train_data, &alpha, &raw_g_x[matrix_start], state.num_attributes, &raw_g_x[I_lo * state.num_attributes], 1, &beta, raw_g_lo_dotprod, 1 );
	
	}

	/*cout << "----------------\n";

	for (int i = 100 ; i < 130; i++) {

		cout << g_lo_dotprod[i] << ",";

	}

	cout << "\n-------------\n";*/

	//printf("G_X_SQ: %x - %x\n", thrust::raw_pointer_cast(&g_x_sq[0]), thrust::raw_pointer_cast(&g_x_sq[state.num_train_data-1]));
	//printf("G_F: %x - %x\n", thrust::raw_pointer_cast(&g_f[0]), thrust::raw_pointer_cast(&g_f[state.num_train_data-1]));
	//printf("G_X_SQ: %x - %x\n", thrust::raw_pointer_cast(&g_x_sq[0]), thrust::raw_pointer_cast(&g_x_sq[state.num_train_data-1]));



	//printf("%x, %x\n", thrust::raw_pointer_cast(&g_hi_dotprod[state.num_attributes-1]), thrust::raw_pointer_cast(&g_lo_dotprod[state.num_attributes-1]));

//	t2 = CycleTimer::currentTicks();
//	cout << "SGEMV 2: " << t2-t1 << "\n";
//	t1 = t2;

	float x_hi_sq = g_x_sq[I_hi];
	float x_lo_sq = g_x_sq[I_lo];
		
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(g_hi_dotprod.begin(), g_lo_dotprod.begin(), g_x_sq.begin()+start, g_f.begin())),
   	                 thrust::make_zip_iterator(thrust::make_tuple(g_hi_dotprod.end(), g_lo_dotprod.end(), g_x_sq.begin()+end, g_f.end())),
       	             update_functor(state.gamma, alpha_lo_old, alpha_hi_old, alpha_lo_new, alpha_hi_new, y_lo, y_hi, x_hi_sq, x_lo_sq));

	/*cout << "----------------\n";

	for (int i = 100 ; i < 130; i++) {
		
		cout << g_f[i] << ",";

	}
	cout << "\n-------------\n";*/
	//prev_hi = I_hi;
	//prev_lo = I_lo;

//	t2 = CycleTimer::currentTicks();
//	cout << "UPDATE_FUNCTOR: " << t2-t1 << "\n";
//	t1 = t2;

/////////////////////////////////////////////////////////


//	t2 = CycleTimer::currentTicks();
//	cout << "Destroy: " << t2-t1 << "\n";
//	t1 = t2;
	return 0;
}

//Parameterized constructor
SvmTrain::SvmTrain(int n_data, int d) {
	num_train_data = n_data;
	start = d;
	end = d+n_data;
	matrix_start = start*state.num_attributes;
	matrix_end = end*state.num_attributes;

	init.I_1 = -1;
	init.I_2 = -1;
	init.f_1 = 1000000000;
	init.f_2 = -1000000000;
}


void SvmTrain::setup(std::vector<float>& raw_x, std::vector<int>& raw_y) {
	
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for DPSVM\n");
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
	
	x = thrust::host_vector<float>(raw_x);
	y = thrust::host_vector<int>(raw_y);

	//cout << "PRE X COPY: \n";
 
	//Copy x and y to device
	g_x = thrust::device_vector<float>(x.begin(), x.end()) ;
	
	//cout << "POST X COPY: \n";
	
	//Initialize alpha on device
	g_alpha = thrust::device_vector<float>(state.num_train_data, 0);
	
	//cout << "POST ALPHA: \n";
	
	init_cuda_handles();
	
	//cout << "POST HANDLE INIT: \n";
	
	g_x_sq = thrust::device_vector<float>(state.num_train_data);
	
	//cout << "POST X_SQ: \n";
	
	for( int i = 0; i < state.num_train_data; i++ )
	{
		g_x_sq[i] = thrust::inner_product(&g_x[i*state.num_attributes], &g_x[i*state.num_attributes] + state.num_attributes, &g_x[i*state.num_attributes], 0.0f);
	}
	
	//cout << "POST X_SQ INIT: \n";

	raw_g_x = thrust::raw_pointer_cast(&g_x[0]);
	
	//cout << "POST G_X: \n";
	
	//ONLY THE FOLLOWING USE INFO PERTAINING TO THIS PARTICULAR SPLIT
	
	g_y = thrust::device_vector<int>(y.begin()+start, y.begin()+end);

	//cout << "POST G_Y: \n";
	
	// Initialize f on device
	g_f  = thrust::device_vector<float>(num_train_data);
	thrust::transform(g_y.begin(), g_y.end(), g_f.begin(), thrust::negate<float>());
	
	//cout << "POST G_F INIT: \n";
	
	lineCache = new myCache(state.cache_size, num_train_data);
	
	//cout << "POST LINECACHE: \n";
	
	rv = new float[4];
	
	g_I_set = thrust::device_vector<i_helper>(num_train_data);
	
	first = thrust::counting_iterator<int>(start);
	last = first + num_train_data;

}
//	t2 = CycleTimer::currentTicks();
	//cout << "POST INIT, PRE G_X_SQ CALC: " << t2 - t1 << "\n";
//	t1 = t2;

struct my_maxmin : public thrust::binary_function<i_helper, i_helper, i_helper> { 

   __host__ __device__
   i_helper operator()(i_helper x, i_helper y) { 
		i_helper rv;//(fminf(x.I_1, y.I_1), fmaxf(x.I_2, y.I_2));
		
		if(x.f_1 < y.f_1) {
			
			rv.I_1 = x.I_1;
			rv.f_1 = x.f_1;

		}
		else { //if (x.f_1 > y.f_1) {
	
			rv.I_1 = y.I_1;
			rv.f_1 = y.f_1;

		}
		/*else {

			if(x.I_1 < y.I_1) {
		
				rv.I_1 = x.I_1;
				rv.f_1 = x.f_1;

			}
			else {
			
				rv.I_1 = y.I_1;
				rv.f_1 = y.f_1;
	
			}				

		}*/



		if(x.f_2 > y.f_2) {
			
			rv.I_2 = x.I_2;
			rv.f_2 = x.f_2;

		}
		else { //if(x.f_2 < y.f_2) {
	
			rv.I_2 = y.I_2;
			rv.f_2 = y.f_2;

		}
		/*else {

			if(x.I_2 < y.I_2) {
		
				rv.I_2 = x.I_2;
				rv.f_2 = x.f_2;

			}
			else {
			
				rv.I_2 = y.I_2;
				rv.f_2 = y.f_2;
	
			}				

		}*/
		return rv; 
	}
};

void SvmTrain::train_step1() {
	
	//Set up I_set1 and I_set2
	thrust::for_each(thrust::make_zip_iterator(thrust::make_tuple(g_alpha.begin() + start, g_y.begin(), g_f.begin(), g_I_set.begin(), first)),
 	                 thrust::make_zip_iterator(thrust::make_tuple(g_alpha.begin() + end, g_y.end(), g_f.end(), g_I_set.end(), last)),
       	             arbitrary_functor(state.c));

	i_helper res  = thrust::reduce(g_I_set.begin(), g_I_set.end(), init, my_maxmin());

	rv[0] = res.I_1;
	rv[1] = res.I_2;
	rv[2] = res.f_1;
	rv[3] = res.f_2;

}

void SvmTrain::train_step2(int I_hi, int I_lo, float alpha_hi_new, float alpha_lo_new) {

	float alpha_lo_old = g_alpha[I_lo];
	float alpha_hi_old = g_alpha[I_hi];
	
	int y_hi = y[I_hi];
	int y_lo = y[I_lo];

	g_alpha[I_lo] = alpha_lo_new;
	g_alpha[I_hi] = alpha_hi_new;

	update_f(I_lo, I_hi, y_lo, y_hi, alpha_lo_old, alpha_hi_old, alpha_lo_new, alpha_hi_new);
}

/*float SvmTrain::get_train_accuracy() {
	int num_correct = 0;

	//thrust::host_vector<float> alpha = g_alpha; 
	//float* raw_alpha = thrust::raw_pointer_cast(&alpha[0]);
	
	for(int i=0; i<state.num_train_data; i++) {
		//cout << "Iter: " << i << "\n";

		cublasSgemv(t_handle, CUBLAS_OP_T, state.num_attributes, new_size, &alpha, &raw_g_x_c[0], state.num_attributes, &raw_g_x[i * state.num_attributes], 1, &beta, raw_g_t_dp, 1 );
	

		float i_sq = g_x_sq[i];

		float dual = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(g_y_c.begin(), g_alpha_c.begin(), g_x_sq_c.begin(), g_t_dp.begin())),
   	                 thrust::make_zip_iterator(thrust::make_tuple(g_y_c.end(), g_alpha_c.end(), g_x_sq_c.end(), g_t_dp.end())),
       	             test_functor<thrust::tuple<int, float, float, float> >(i_sq), 0, thrust::plus<float>());
		

		//dual += y[j]*raw_alpha[j]*rbf_kernel(j,i);
		//	}
		//}

		dual += b;

		int result = 1;
		if(dual < 0) {
			result = -1;
		}

		if(result == y[i]) {
			num_correct++;
		}
	}

	return ((float)num_correct/(state.num_train_data));
}*/


struct is_not_sv
{
  template <typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t)
  {
    return (thrust::get<0>(t) <= 0);
  }
};

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


void SvmTrain::test_setup() {

	g_alpha_c = g_alpha;
	g_y_c = y;
	g_x_sq_c = g_x_sq;
	g_sv_indices = thrust::device_vector<int>(state.num_train_data);

	thrust::sequence(g_sv_indices.begin(), g_sv_indices.end());

	aggregate_sv();

	g_t_dp = thrust::device_vector<float>(new_size);
	raw_g_t_dp = thrust::raw_pointer_cast(&g_t_dp[0]);

	cublasStatus_t status;
	status = cublasCreate(&t_handle);
	
	if (status != CUBLAS_STATUS_SUCCESS) { 

		cout << "CUBLAS initialization failed\n"; 
		exit(EXIT_FAILURE); 
	}

}


void SvmTrain::aggregate_sv() {

	new_size = thrust::remove_if(thrust::device, 
					  thrust::make_zip_iterator(thrust::make_tuple(g_alpha_c.begin(), g_y_c.begin(), g_x_sq_c.begin(), g_sv_indices.begin())), 
					  thrust::make_zip_iterator(thrust::make_tuple(g_alpha_c.end(), g_y_c.end(), g_x_sq_c.end(), g_sv_indices.end())),
					  is_not_sv()) 
					  - thrust::make_zip_iterator(thrust::make_tuple(g_alpha_c.begin(), g_y_c.begin(), 
																	g_x_sq_c.begin(), g_sv_indices.begin()));

	cout << "Number of SVs: " << new_size << "\n";

	g_alpha_c.resize(new_size);
	g_y_c.resize(new_size);
	g_x_sq_c.resize(new_size);
	g_sv_indices.resize(new_size);	
	

	thrust::host_vector<int> temp_indices = g_sv_indices; 
	thrust::host_vector<float> temp_x(new_size * state.num_attributes);

	for(int i = 0 ; i < new_size; i++) {

		int idx = temp_indices[i];		

		for(int j = 0; j < state.num_attributes; j++){

			temp_x[i*state.num_attributes + j] = x[idx*state.num_attributes + j];
	
		}

	}

	g_x_c = temp_x;

	raw_g_x_c = thrust::raw_pointer_cast(&g_x_c[0]);

}

float SvmTrain::get_train_accuracy() {
	int num_correct = 0;

	
	for(int i=0; i<state.num_train_data; i++) {

	
		cublasSgemv(t_handle, CUBLAS_OP_T, state.num_attributes, new_size, &alpha, &raw_g_x_c[0], state.num_attributes, &raw_g_x[i * state.num_attributes], 1, &beta, raw_g_t_dp, 1 );


		float i_sq = g_x_sq[i];

	
		float dual = 0.0f;

		dual = thrust::transform_reduce(thrust::make_zip_iterator(thrust::make_tuple(g_y_c.begin(), g_alpha_c.begin(), g_x_sq_c.begin(), g_t_dp.begin())),
   	                 thrust::make_zip_iterator(thrust::make_tuple(g_y_c.end(), g_alpha_c.end(), g_x_sq_c.end(), g_t_dp.end())),
       	             test_functor<thrust::tuple<int, float, float, float> >(i_sq, state.gamma), 0.0f, thrust::plus<float>());
		
		dual -= b;

		int result = 1;
		if(dual < 0.0f) {
			result = -1;
		}

		if(result == y[i]) {
			num_correct++;
		}
	}

	return ((float)num_correct/(state.num_train_data));
}


void SvmTrain::destroy_t_cuda_handles() {

	cublasDestroy(t_handle);

}

float SvmTrain::clip_value(float num, float low, float high) {
	if(num < low) {
		return low;
	} else if(num > high) {
		return high;
	}

	return num;
}

void SvmTrain::get_x(float* x, float* x_copy, int idx, int num_attributes) {
	int ctr = 0;

	int start_index = (idx*num_attributes);
	int end_index = start_index+num_attributes;

	for(int i = start_index; i < end_index; i++) {
		x_copy[ctr++] = x[i];
	}
}


float SvmTrain::rbf_kernel(int i1, int i2){
	
	float* i2_copy = new float[state.num_attributes];

	float* raw_i1 = thrust::raw_pointer_cast(&x[i1*state.num_attributes]);
	float* raw_i2 = thrust::raw_pointer_cast(&x[i2*state.num_attributes]);

	get_x(raw_i2, i2_copy, 0, state.num_attributes);
	
	cblas_saxpy(state.num_attributes, -1, raw_i1, 1, i2_copy, 1); 

	float norm_sq = cblas_sdot(state.num_attributes, i2_copy, 1, i2_copy, 1);

	float result = (float)exp(-1 *(float)state.gamma*norm_sq);

	delete [] i2_copy;

	return result;
}
