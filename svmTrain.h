#ifndef SVMTRAIN
#define SVMTRAIN

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cblas.h>
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
#include <map>
#include <list>
#include <vector>
#include <iostream>

struct i_h_def{

	int I_1;
	int I_2;
	float f_1;
	float f_2;

	//i_h_def () : I_1(-1), I_2(-1), f_1(1000000000), f_2(-1000000000) {}
	
	//i_h_def (float f1, float f2) : I_1(-1), I_2(-1), f_1(f1), f_2(f2) {}

};

typedef struct i_h_def i_helper;

typedef struct {

	int I_hi;
	int I_lo;
	float b_hi;
	float b_lo;

} step1_rv;

class SvmTrain {

	private:

		thrust::host_vector<float> x;
		thrust::host_vector<int> y;

		thrust::device_vector<float> g_x;
		thrust::device_vector<int> g_y;
			
		thrust::device_vector<float> g_f;

		thrust::device_vector<float> g_alpha;
		
	 
		thrust::device_vector<float> g_x_sq;
	
		thrust::counting_iterator<int> first;
		thrust::counting_iterator<int> last;

		thrust::device_vector<i_helper> g_I_set;

		float* raw_g_x;

		cublasHandle_t handle;
		cudaStream_t stream1;
		cudaStream_t stream2;

		//Cache for kernel computations
		myCache* lineCache;

		int num_train_data;
		int disp;
		int start;
		int end;
		int matrix_start;
		int matrix_end;
	
		i_helper init;

	//////////////// TEST RELATED ////////////

	
		thrust::device_vector<float> g_alpha_c;
		thrust::device_vector<float> g_x_c;
		thrust::device_vector<int> g_y_c;
		thrust::device_vector<float> g_x_sq_c;
		thrust::device_vector<float> g_t_dp;
		thrust::device_vector<int> g_sv_indices;


		float* raw_g_x_c;
		float* raw_g_t_dp;
		int new_size;
	
		cublasHandle_t t_handle;
	/////////////////////////////////////////

	public:
	
		float* rv;
		float b;

		SvmTrain(int n_data, int d);

		void setup(std::vector<float>& raw_x, std::vector<int>& raw_y);

		void train_step1();

		void train_step2(int I_hi, int I_lo, float alpha_hi, float alpha_lo);

		void init_cuda_handles();

		void destroy_cuda_handles();
		
		int update_f(int I_lo, int I_hi, int y_lo, int y_hi, float alpha_lo_old, float alpha_hi_old, float alpha_lo_new, float alpha_hi_new);

		thrust::device_vector<float>& lookup_cache(int I_idx, bool& cache_hit);

		float clip_value(float num, float low, float high);
		
		float rbf_kernel(int i1, int i2);
		
		void get_x(float* x, float* x_copy, int idx, int num_attributes);

		float get_train_accuracy();

		void test_setup();

		void aggregate_sv();
	
		void destroy_t_cuda_handles();
};

#endif
