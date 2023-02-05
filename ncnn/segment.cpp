
#include "net.h"
#include "mat.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <omp.h>

#include <iostream>
#include <random>
#include <algorithm>
#include <stdio.h>
#include <string>
#include <vector>


using std::string;
using std::vector;
using cv::Mat;


vector<vector<uint8_t>> get_color_map();
void inference();


int main(int argc, char** argv) {
    inference();
    return 0;
}


void inference() {
    int nthreads = 4;
    string mod_param = "../models/model_v2_sim.param";
    string mod_model = "../models/model_v2_sim.bin";
    int oH{512}, oW{1024}, n_classes{19};
    float mean[3] = {0.3257f, 0.3690f, 0.3223f};
    float var[3] = {0.2112f, 0.2148f, 0.2115f};
    string impth = "../../example.png";
    string savepth = "out.png";
    
    // load model
    ncnn::Net mod;
#if NCNN_VULKAN
    int gpu_count = ncnn::get_gpu_count();
    if (gpu_count <= 0) {
        fprintf(stderr, "we do not have gpu device\n");
        return;
    }
    mod.opt.use_vulkan_compute = 1;
    mod.set_vulkan_device(1);
#endif 
    //// switch off fp16
    // bool use_fp16 = false;
    // mod.opt.use_fp16_packed = use_fp16;
    // mod.opt.use_fp16_storage = use_fp16;
    // mod.opt.use_fp16_arithmetic = use_fp16;
    //// switch on bf16
    // mod.opt.use_packing_layout = true;
    // mod.opt.use_ff16_storage = true;
    //// reduce cpu usage
    // net.opt.openmp_blocktime = 0; 
    mod.opt.use_winograd_convolution = true;

    // we should set opt before load model
    mod.load_param(mod_param.c_str());
    mod.load_model(mod_model.c_str());

    // load image, and copy to ncnn mat
    cv::Mat im = cv::imread(impth);
    if (im.empty()) {
        fprintf(stderr, "cv::imread failed\n");
        return;
    }

    ncnn::Mat inp = ncnn::Mat::from_pixels_resize(
            im.data, ncnn::Mat::PIXEL_BGR, im.cols, im.rows, oW, oH);
    for (float &el : mean) el *= 255.;
    for (float &el : var) el = 1. / (255. * el); 
    inp.substract_mean_normalize(mean, var);

    // set input, run, get output
    ncnn::Extractor ex = mod.create_extractor();
    ex.set_light_mode(true); 
    ex.set_num_threads(nthreads);
#if NCNN_VULKAN
    ex.set_vulkan_compute(true);
#endif

    ex.input("input_image", inp);
    ncnn::Mat out;
    ex.extract("preds", out); // output is nchw, as onnx, where here n=1

    // generate colorful output, and dump
    vector<vector<uint8_t>> color_map = get_color_map();
    Mat pred(cv::Size(oW, oH), CV_8UC3);
    int offset = oH * oW;
    omp_set_num_threads(omp_get_max_threads());
    #pragma omp parallel for 
    for (int i=0; i < oH; ++i) {
        uint8_t *ptr = pred.ptr<uint8_t>(i);
        for (int j{0}; j < oW; ++j) {
            // compute argmax
            int idx, argmax{0}; 
            float max;
            idx = i * oW + j;
            max = out[idx];
            for (int k{1}; k < n_classes; ++k) {
                idx += offset;
                if (max < out[idx]) {
                    max = out[idx];
                    argmax = k;
                }
            }
            // color the result
            ptr[0] = color_map[argmax][0];
            ptr[1] = color_map[argmax][1];
            ptr[2] = color_map[argmax][2];
            ptr += 3;
        }
    }
    cv::imwrite(savepth, pred);

    ex.clear(); // must have this, or error
    mod.clear();

}


vector<vector<uint8_t>> get_color_map() {
    vector<vector<uint8_t>> color_map(256, vector<uint8_t>(3));
    std::minstd_rand rand_eng(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (int i{0}; i < 256; ++i) {
        for (int j{0}; j < 3; ++j) {
            color_map[i][j] = u(rand_eng);
        }
    }
    return color_map;
}
