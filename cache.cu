#include "cache.hpp"
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
	}   
}

thrust::device_vector<float>* myCache::lookup(int key) {

	std::map<int,int>::iterator it = my_map.find(key);

	if(it != my_map.end()) {

		order.remove(key);
		order.push_back(key);

		return &lines[it->second];

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

		my_map.erase(it);

		my_map[key] = line_number;

		order.push_back(key);
		order.pop_front();

		return lines[line_number];
	}

	my_map[key] = size;
	order.push_back(key);

	return lines[size++];

}
