#ifndef SVMTRAIN_HPP
#define SVMTRAIN_HPP

typedef struct {

	int num_attributes;
	int num_train_data;
	float c;
	float gamma;
	float epsilon;
	char input_file_name[200];
	char model_file_name[200];
	int max_iter;
	int cache_size;

} state_model;

//global structure for training parameters
extern state_model state;
#endif
