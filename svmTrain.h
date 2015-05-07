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
		
		thrust::device_vector<float> g_x_hi;
		thrust::device_vector<float> g_x_lo;
		
		thrust::device_vector<float> g_f;

		thrust::device_vector<float> g_alpha;
		
	 
		thrust::device_vector<float> g_x_sq;

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

	public:
	
		float b;

		SvmTrain(int n_data, int d);

		void setup(std::vector<float>& raw_x, std::vector<int>& raw_y);

		step1_rv train_step1();

		void train_step2(int I_hi, int I_lo, float alpha_hi, float alpha_lo);

		void init_cuda_handles();

		void destroy_cuda_handles();
		
		int update_f(int I_lo, int I_hi, int y_lo, int y_hi, float alpha_lo_old, float alpha_hi_old, float alpha_lo_new, float alpha_hi_new);

		thrust::device_vector<float>& lookup_cache(int I_idx, bool& cache_hit);

		float clip_value(float num, float low, float high);
		
		float rbf_kernel(int i1, int i2);
		
		void get_x(float* x, float* x_copy, int idx, int num_attributes);

		float get_train_accuracy();

};

#endif
