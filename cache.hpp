#ifndef CACHE_H
#define CACHE_H

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

#endif