#ifndef _TRT_DEP_HPP_
#define _TRT_DEP_HPP_

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include "NvInferPlugin.h"
#include <cuda_runtime_api.h>

#include <iostream>
#include <string>
#include <vector>
#include <memory>


using std::string;
using std::vector;
using std::cout;
using std::endl;

using nvinfer1::ICudaEngine;
using nvinfer1::ILogger;
using Severity = nvinfer1::ILogger::Severity;


class Logger: public ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity != Severity::kINFO) {
                std::cout << msg << std::endl;
            }
        }
};

struct TrtDeleter {
    template <typename T>
    void operator()(T* obj) const {
        delete obj; 
    }
};

struct CudaStreamDeleter {
    void operator()(cudaStream_t* stream) const {
        cudaStreamDestroy(*stream);
    }
};

template <typename T>
using TrtUnqPtr = std::unique_ptr<T, TrtDeleter>;
using CudaStreamUnqPtr = std::unique_ptr<cudaStream_t, CudaStreamDeleter>;
using TrtSharedEnginePtr = std::shared_ptr<ICudaEngine>;


extern Logger gLogger;


TrtSharedEnginePtr shared_engine_ptr(ICudaEngine* ptr);
TrtSharedEnginePtr parse_to_engine(string onnx_path, string quant, 
        string data_root, string data_file);
void serialize(TrtSharedEnginePtr engine, string save_path);
TrtSharedEnginePtr deserialize(string serpth);
vector<int> infer_with_engine(TrtSharedEnginePtr engine, vector<float>& data);
void test_fps_with_engine(TrtSharedEnginePtr engine);

#endif
