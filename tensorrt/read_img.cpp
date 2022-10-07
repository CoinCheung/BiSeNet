
#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <sstream>
#include <stdexcept>


using std::cout;
using std::endl;
using std::vector;
using std::string;
using cv::Mat;


void read_data(std::string impth, float *data, int iH, int iW, 
        int& orgH, int& orgW) {
    vector<float> mean{0.485f, 0.456f, 0.406f}; // rgb order
    vector<float> variance{0.229f, 0.224f, 0.225f};

    Mat im = cv::imread(impth);
    if (im.empty()) {
        cout << "cannot read image \n";
        std::abort();
    }

    orgH = im.rows; orgW = im.cols;
    if ((orgH != iH) || orgW != iW) {
        cout << "resize orignal image of (" << orgH << "," << orgW 
            << ") to (" << iH << ", " << iW << ") according to model require\n";
        cv::resize(im, im, cv::Size(iW, iH), cv::INTER_CUBIC);
    }

    // normalize and convert to rgb
    float scale = 1.f / 255.f;
    for (int i{0}; i < variance.size(); ++ i) {
        variance[i] = 1.f / variance[i];
    }
    for (int h{0}; h < iH; ++h) {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w < iW; ++w) {
            for (int c{0}; c < 3; ++c) {
                int idx = c * iH * iW + h * iW + w; 
                data[idx] = (p[w][2 - c] * scale - mean[c]) * variance[c];
            }
        }
    }
}


void read_data(std::string impth, float *data, int iH, int iW) {
    int tmp1, tmp2;
    read_data(impth, data, iH, iW, tmp1, tmp2);
}

