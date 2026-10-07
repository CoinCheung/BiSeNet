
#ifndef _READ_IMAGE_HPP_
#define _READ_IMAGE_HPP_


#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <cstdint>


using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;


void read_data(std::string impth, uint8_t *data,
        int iH, int iW, int& orgH, int& orgW);
void read_data(std::string impth, uint8_t *data, int iH, int iW);


#endif 
