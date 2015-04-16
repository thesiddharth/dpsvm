#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cblas.h>
#include <math.h>

using namespace std;

void populate_data(char* input_file_name, float* x, int* y, int num_attributes, int num_train_data);
void initialize_f_array(float* f, int* y, int num_train_data);
void set_I_arrays(float* alpha, int *y, float c, int num_train_data, vector<int> I[5]);
int get_I_up(float* f, vector<int> I[5]);
int get_I_low(float* f, vector<int> I[5]);
void get_x(float* x, float* x_copy, int idx, int num_attributes);
float rbf_kernel(float* x1,float* x2,float gamma,int num_attributes);
float clip_value(float num, float low, float high);
void update_f(float* f, float* x, float* x1, float* x2, int y1, int y2, float d_alpha1, float d_alpha2, int num_train_data, int num_attributes);

int main(int argc, char *argv[]) {
	//TODO: command line arguments here
	int num_attributes = 120;
	int num_train_data = 10000;
	float C = 1;
	float gamma = 0.1;
	char input_file_name[30] = "train.csv";

	//input data attributes and labels
	float* x = new float[num_attributes*num_train_data];
	int* y = new int[num_train_data];

	populate_data(input_file_name, x, y, num_attributes, num_train_data);

	//Initialize starting values
	float dual = 0;
	float* alpha = new float[num_train_data]();
	float* f = new float[num_train_data];

	initialize_f_array(f, y, num_train_data);

	vector<int> I[5];

	set_I_arrays(alpha, y, C, num_train_data, I);

	int I_up = get_I_up(f,I);
	float b_up = f[I_up];

	int I_low = get_I_low(f,I);
	float b_low = f[I_low];

	float* x1 = new float[num_attributes];
	float* x2 = new float[num_attributes];

	get_x(x, x1, I_low, num_attributes);
	get_x(x, x2, I_up, num_attributes);

	int y1 = y[I_low];
	int y2 = y[I_up];

	int s = y1*y2;
	float eta = (2*rbf_kernel(x1,x2,gamma,num_attributes)) - rbf_kernel(x1,x1,gamma,num_attributes) - rbf_kernel(x2,x2,gamma,num_attributes);

	float alpha1_old = alpha[I_low];
	float alpha2_old = alpha[I_up];

	//update alpha1 and alpha2
	float alpha2_new = alpha2_old - (y2*(b_low - b_up)/eta);
	float alpha1_new = alpha1_old + (s*(alpha2_old - alpha2_new));

	//clip alpha values between 0 and C
	alpha1_new = clip_value(alpha1_new, 0, C);
	alpha2_new = clip_value(alpha2_new, 0, C);

	//store new alpha_1 and alpha_2 values
	alpha[I_low] = alpha1_new;
	alpha[I_up] = alpha2_new;

	update_f(f, x, x1, x2, y1, y2, (alpha1_new - alpha1_old), (alpha2_new - alpha2_old), num_train_data, num_attributes);

	return 0;
}

void update_f(float* f, float* x, float* x1, float* x2, int y1, int y2, float d_alpha1, float d_alpha2, int num_train_data, int num_attributes) {
	for(int i=0; i<num_train_data; i++) {
		float* xi = new float[num_attributes];
		get_x(x, xi, i, num_attributes);

		float delta = (d_alpha1*y1*rbf_kernel(x1,xi)) + (d_alpha2*y2*rbf_kernel(x2,xi));

		f[i] += delta;
	}
}

float clip_value(float num, float low, float high) {
	if(num < low) {
		return low;
	} else if(num > high) {
		return high;
	}

	return num;
}

float rbf_kernel(float* x1,float* x2,float gamma,int num_attributes) {
	float* x1_copy = new float[num_attributes];
	float* x2_copy = new float[num_attributes];

	//deep copy
	get_x(x1, x1_copy, 0, num_attributes);
	get_x(x2, x2_copy, 0, num_attributes);

	cblas_saxpy(num_attributes, -1, x2_copy, 1, x1_copy, 1); // x1_copy = -x2_copy + x1_copy

	float norm = cblas_snrm2(num_attributes, x1_copy, 1);

	float result = (float)exp((double)gamma*norm*norm);

	return result;
}

void get_x(float* x, float* x_copy, int idx, int num_attributes) {
	int ctr = 0;

	for(int i=(idx*num_attributes); i<num_attributes; i++) {
		x_copy[ctr++] = x[i];
	}
}

void populate_data(char* input_file_name, float* x, int* y, int num_attributes, int num_train_data)
{
    ifstream file(input_file_name);

    if(!file.is_open())
    {
        cout << "Couldn't open file";
        return;
    }
    //std::vector<std::string>   result;
    string line;
    int curr_example_num = 0;

    while (curr_example_num < num_train_data)
    {
        getline(file,line);

        stringstream lineStream(line);
        string cell;

        getline(lineStream,cell,',');

        y[curr_example_num] = stoi(cell);

        int curr_attr_num = 0;

        while(getline(lineStream,cell,','))
        {
            x[(curr_example_num * num_attributes) + curr_attr_num++] = stof(cell);
        }

        ++curr_example_num;
    }
}

void initialize_f_array(float* f, int* y, int num_train_data) {
	for(int i=0; i<num_train_data; i++) {
		f[i] = -1*y[i];
	}
}

void set_I_arrays(float* alpha, int* y, float c, int num_train_data, vector<int> I[5]) {
	//clear vectors before populating
	for(int i=0; i<5; ++i) {
		I[i].clear();
	}
		
	//populate the I sets
	for(int i=0; i<num_train_data; ++i) {
		if(alpha[i] == 0) {
			if(y[i] == 1) {
				I[1].push_back(i);
			} else {
				I[4].push_back(i);
			}
		} else if(alpha[i] == c) {
			if(y[i] == -1) {
				I[2].push_back(i);
			} else {
				I[3].push_back(i);
			}
		} else {
			I[0].push_back(i);
		}
	}
}

int get_I_up(float* f, vector<int> I[5]) {
	int I_up = 0;
	float min = 1000000000;

	for(vector<int>::iterator it = I[0].begin(); it != I[0].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}

	for(vector<int>::iterator it = I[1].begin(); it != I[1].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}

	for(vector<int>::iterator it = I[2].begin(); it != I[2].end(); ++it) {
		if(f[*it] < min) {
			min = f[*it];
			I_up = *it;
		}
	}
		
	return I_up;
}

int get_I_low(float* f, vector<int> I[5]) {
	int I_low = 0;
    float max = -1000000000;

    for(vector<int>::iterator it = I[0].begin(); it != I[0].end(); ++it) {
		if(f[*it] > max) {
			max = f[*it];
			I_low = *it;
		}
	}

	for(vector<int>::iterator it = I[3].begin(); it != I[3].end(); ++it) {
		if(f[*it] > max) {
			max = f[*it];
			I_low = *it;
		}
	}

	for(vector<int>::iterator it = I[4].begin(); it != I[4].end(); ++it) {
		if(f[*it] > max) {
			max = f[*it];
			I_low = *it;
		}
	}
	
	return I_low;
}
