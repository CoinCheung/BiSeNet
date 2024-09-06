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
using nvinfer1::IRuntime;
using Severity = nvinfer1::ILogger::Severity;


void CHECK(bool success, string msg);


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


struct SemanticSegmentTrt {
public:
    TrtSharedEnginePtr engine;
    CudaStreamUnqPtr stream;
    TrtUnqPtr<IRuntime> runtime;

    string input_name;
    string output_name;
    int opt_bsize{1};

    SemanticSegmentTrt(): 
        engine(nullptr), runtime(nullptr), stream(nullptr) {

        stream.reset(new cudaStream_t);
        auto fail = cudaStreamCreate(stream.get());
        CHECK(!fail, "create stream failed");
    }

    ~SemanticSegmentTrt() {
        engine.reset();
        runtime.reset();
        stream.reset();
    }

    void set_opt_batch_size(int bs);

    void serialize(string save_path);

    void deserialize(string serpth);

    void parse_to_engine(string onnx_path, string quant, 
        string data_root, string data_file);

    vector<int> inference(vector<float>& data);

    void test_speed_fps();

    vector<int> get_input_shape();
    vector<int> get_output_shape();
};


#endif
