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


// Raw HWC uint8 pixels (/255, mean/std, layout are in the onnx); only BGR->RGB here.
void read_data(std::string impth, uint8_t *data, int iH, int iW,
        int& orgH, int& orgW) {

    Mat im = cv::imread(impth);
    if (im.empty()) {
        cout << "cannot read image \n";
        std::abort();
    }

    orgH = im.rows; orgW = im.cols;
    if ((orgH != iH) || orgW != iW) {
        cout << "resize orignal image of (" << orgH << "," << orgW
            << ") to (" << iH << ", " << iW << ") according to model require\n";
        cv::resize(im, im, cv::Size(iW, iH), 0, 0, cv::INTER_LINEAR);
    }

    for (int h{0}; h < iH; ++h) {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w < iW; ++w) {
            uint8_t *o = data + (h * iW + w) * 3;
            o[0] = p[w][2];   // R
            o[1] = p[w][1];   // G
            o[2] = p[w][0];   // B
        }
    }
}


void read_data(std::string impth, uint8_t *data, int iH, int iW) {
    int tmp1, tmp2;
    read_data(impth, data, iH, iW, tmp1, tmp2);
}
