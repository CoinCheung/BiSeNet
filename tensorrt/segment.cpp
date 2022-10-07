#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>
#include "NvInferRuntimeCommon.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <random>

#include "trt_dep.hpp"
#include "read_img.hpp"


using nvinfer1::IHostMemory;
using nvinfer1::IBuilder;
using nvinfer1::INetworkDefinition;
using nvinfer1::ICudaEngine;
using nvinfer1::IInt8Calibrator;
using nvinfer1::IBuilderConfig;
using nvinfer1::IRuntime;
using nvinfer1::IExecutionContext;
using nvinfer1::ILogger;
using nvinfer1::Dims3;
using nvinfer1::Dims2;
using Severity = nvinfer1::ILogger::Severity;

using std::string;
using std::ios;
using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;
using std::array;

using cv::Mat;




vector<vector<uint8_t>> get_color_map();

void compile_onnx(vector<string> args);
void run_with_trt(vector<string> args);
void test_speed(vector<string> args);


int main(int argc, char* argv[]) {
    if (argc < 3) {
        cout << "usage is ./segment compile/run/test\n";
        std::abort();
    }

    vector<string> args;
    for (int i{1}; i < argc; ++i) args.emplace_back(argv[i]);

    if (args[0] == "compile") {
        if (argc < 4) {
            cout << "usage is: ./segment compile input.onnx output.trt [--fp16|--fp32]\n";
            cout << "or ./segment compile input.onnx output.trt --int8 /path/to/data_root /path/to/ann_file\n";
            std::abort();
        }
        compile_onnx(args);
    } else if (args[0] == "run") {
        if (argc < 5) {
            cout << "usage is ./segment run ./xxx.trt input.jpg result.jpg\n";
            std::abort();
        }
        run_with_trt(args);
    } else if (args[0] == "test") {
        if (argc < 3) {
            cout << "usage is ./segment test ./xxx.trt\n";
            std::abort();
        }
        test_speed(args);
    }

    return 0;
}


void compile_onnx(vector<string> args) {
    string quant("fp32");
    string data_root("none");
    string data_file("none");
    if ((args.size() >= 4)) {
        if (args[3] == "--fp32") {
            quant = "fp32";
        } else if (args[3] == "--fp16") {
            quant = "fp16";
        } else if (args[3] == "--int8") {
            quant = "int8";
            data_root = args[4];
            data_file = args[5];
        } else {
            cout << "invalid args of quantization: " << args[3] << endl; 
            std::abort();
        }
    } 

    TrtSharedEnginePtr engine = parse_to_engine(args[1], quant, data_root, data_file);
    serialize(engine, args[2]);
}


void run_with_trt(vector<string> args) {

    TrtSharedEnginePtr engine = deserialize(args[1]);

    Dims3 i_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    Dims3 o_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));
    const int iH{i_dims.d[2]}, iW{i_dims.d[3]};
    const int oH{o_dims.d[2]}, oW{o_dims.d[3]};

    // prepare image and resize
    vector<float> data; data.resize(iH * iW * 3);
    int orgH, orgW;
    read_data(args[2], &data[0], iH, iW, orgH, orgW);

    // call engine
    vector<int> res = infer_with_engine(engine, data);

    // generate colored out
    vector<vector<uint8_t>> color_map = get_color_map();
    Mat pred(cv::Size(oW, oH), CV_8UC3);
    int idx{0};
    for (int i{0}; i < oH; ++i) {
        uint8_t *ptr = pred.ptr<uint8_t>(i);
        for (int j{0}; j < oW; ++j) {
            ptr[0] = color_map[res[idx]][0];
            ptr[1] = color_map[res[idx]][1];
            ptr[2] = color_map[res[idx]][2];
            ptr += 3;
            ++idx;
        }
    }

    // resize back and save
    if ((orgH != oH) || (orgW != oW)) {
        cv::resize(pred, pred, cv::Size(orgW, orgH), cv::INTER_CUBIC);
    }
    cv::imwrite(args[3], pred);
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


void test_speed(vector<string> args) {
    TrtSharedEnginePtr engine = deserialize(args[1]);
    test_fps_with_engine(engine);
}
