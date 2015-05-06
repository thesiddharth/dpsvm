#ifndef SVMTRAIN
#define SVMTRAIN

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

class myCache {

	private:
			
			int line_size;	
			int max_size;
			int size;
			std::vector< thrust::device_vector<float> > lines;
			std::map<int, int> my_map; 
			std::list<int> order;

	public:

			void dump_map_contents();
			
			myCache(int max_size, int line_size);

			thrust::device_vector<float>* lookup(int key);

			thrust::device_vector<float>& get_new_cache_line(int key);
};



void myCache::dump_map_contents() {

	std::map<int,int>::iterator it;

	std::cout << "--------\n";

	for(it = my_map.begin(); it != my_map.end(); ++it) {

		std::cout << it->first << "," << it->second << "::" ;

	}
	
	std::cout << "\n";
	
	std::list<int>::iterator it2;
	
	for(it2 = order.begin(); it2 != order.end(); ++it2) {

		std::cout << *it2 << "::" ;

	}

	std::cout << "\n----------\n";


}


myCache::myCache(int max_size, int line_size) {

	this->max_size = max_size;	
	this->line_size = line_size;
	this->size = 0;
	lines.resize(max_size);

	for(int i = 0; i < max_size; i++) {

		lines[i] = thrust::device_vector<float>(line_size);
		//line = thrust::device_vector<float>(line_size);

	}

}

thrust::device_vector<float>* myCache::lookup(int key) {

	//std::cout << "Looking up " << key << "\n";

	std::map<int,int>::iterator it = my_map.find(key);

	if(it != my_map.end()) {

		order.remove(key);
		order.push_back(key);

		return &lines[it->second];

		//return &line;

	}
	else {

		return NULL;

	}

}

thrust::device_vector<float>& myCache::get_new_cache_line(int key) {

	if(size == max_size){

		int del_key = order.front();
		std::map<int,int>::iterator it = my_map.find(del_key);
		int line_number = it->second;

//		thrust::fill(this->lines[line_number].begin(), this->lines[line_number].end(), 0);

		my_map.erase(it);

		my_map[key] = line_number;

		order.push_back(key);
		order.pop_front();	

		//std::cout << "Returning new line: " << line_number << "\n";
		return lines[line_number];
	}

	my_map[key] = size;
	order.push_back(key);
		
	//std::cout << "Returning new line: " << size << "\n";
	return lines[size++];
	
}



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

public:
	
	float b_lo;
	float b_hi;
	float b;

    SvmTrain();

    void setup(std::vector<float>& raw_x, std::vector<int>& raw_y);

    void train_step();

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
