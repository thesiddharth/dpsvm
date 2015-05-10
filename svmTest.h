#ifndef SVMTEST
#define SVMTEST

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda.h>
#include <cblas.h>

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

class SvmTest {
	private:
		//test data
		thrust::host_vector<float> x;
		thrust::host_vector<int> y;

		thrust::device_vector<float> g_x;
		thrust::device_vector<int> g_y;

		thrust::device_vector<float> g_x_sq;

		int num_test_data;
		int num_attributes;

		//model data
		thrust::host_vector<float> x_model;
		thrust::host_vector<int> y_model;

		thrust::device_vector<float> g_x_model;
		thrust::device_vector<int> g_y_model;

		thrust::host_vector<float> alpha;
		thrust::device_vector<float> g_alpha;

		thrust::device_vector<float> g_x_sq_model;

		int num_sv;
		float b;
		float gamma;

		//cuda handles
		cublasHandle_t handle;
		cudaStream_t stream1;
		cudaStream_t stream2;

		//////////////// TEST RELATED ////////////
		thrust::device_vector<float> g_t_dp;


		float* raw_g_x;
		float* raw_g_x_c;
		float* raw_g_t_dp;
		/////////////////////////////////////////

	public:
		SvmTest(float model_b, int model_num_sv, int num_test_data_ip, int model_num_attributes, float model_gamma);
		void setup(std::vector<float>& raw_x, std::vector<int>& raw_y, std::vector<float>& raw_x_model, std::vector<int>& raw_y_model, std::vector<float>& raw_alpha);

		float get_test_accuracy();

		void init_cuda_handles();
		void destroy_cuda_handles();
};

#endif
