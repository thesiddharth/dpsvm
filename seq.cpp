#include <stdio.h>
#include <cblas.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

using namespace std;

void populate_data(char* input_file_name, float* x, int* y, int num_attributes, int num_train_data);
void initialize_f_array(float* f, int* y, int num_train_data);
void set_I_arrays(float* alpha, int *y, float c, int num_train_data, vector<int> I[5]);
int get_I_up(float* f, vector<int> I[5]);
int get_I_low(float* f, vector<int> I[5]);

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

	return 0;
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
