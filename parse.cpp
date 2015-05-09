#include "parse.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace std;

void populate_data(vector<float> &x, vector<int> &y, int num_train_data, int num_attributes,char input_file_name[200] ) {
	
    ifstream file(input_file_name);

    if(!file.is_open())
    {
        cout << "Couldn't open file\n";
        exit(-1);
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
