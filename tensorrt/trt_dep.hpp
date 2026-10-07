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
using nvinfer1::IExecutionContext;
using nvinfer1::ILogger;
using nvinfer1::IRuntime;
using Severity = nvinfer1::ILogger::Severity;


void CHECK(bool success, string msg);
void CHECK_CUDA(cudaError_t state, string msg);


class Logger: public ILogger {
    public:
        void log(Severity severity, const char* msg) noexcept override {
            if (severity < Severity::kINFO) {
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
        delete stream;
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

    // profile batch upper bound, used by set_opt_batch_size and parse_to_engine
    static constexpr int kMaxBatch = 4;

    string input_name;
    string output_name;
    int opt_bsize{1};

    // persistent execution state, reused across calls and captured into a cuda graph
    TrtUnqPtr<IExecutionContext> context{nullptr};
    void* d_input{nullptr};
    void* d_output{nullptr};
    // pinned host staging buffers; read_data writes straight into h_input
    void* h_input{nullptr};
    void* h_output{nullptr};     // out_numel elements of out_dtype
    bool use_pinned{true};
    int64_t in_numel{0};    // elements of the input buffer
    int in_elem_size{0};    // bytes per input element, from the engine's dtype
    int64_t out_numel{0};   // elements of the output buffer
    int out_elem_size{0};   // bytes per output element: 4 int32, 1 uint8, 8 int64
    nvinfer1::DataType out_dtype{nvinfer1::DataType::kINT32};
    int ctx_bsize{0};       // batch size the context/graph is bound to, 0 = not ready
    cudaGraphExec_t graph_exec{nullptr};

    SemanticSegmentTrt():
        engine(nullptr), runtime(nullptr), stream(nullptr) {

        cudaStream_t s{};
        CHECK(cudaStreamCreate(&s) == cudaSuccess, "create stream failed");
        stream.reset(new cudaStream_t{s});   // the deleter only ever sees a created stream
        // the fused head plugin self-registers at static init, nothing to do here
    }

    ~SemanticSegmentTrt() {
        release_context();
        engine.reset();
        runtime.reset();
        stream.reset();
    }

    // allocate buffers, bind a context to batch size bs and capture a cuda graph
    void setup_context(int bs);
    // one inference on *stream: graph replay if captured, plain enqueue otherwise
    void launch_once();
    void release_context();

    void set_opt_batch_size(int bs);

    void serialize(string save_path);

    void deserialize(string serpth);

    void parse_to_engine(string onnx_path);

    // the caller fills input_buffer() before calling inference()
    uint8_t* input_buffer() { return static_cast<uint8_t*>(h_input); }

    // H2D + engine + D2H; the labels are then in output_buffer(), typed by output_dtype()
    void infer();
    const void* output_buffer() const { return h_output; }
    nvinfer1::DataType output_dtype() const { return out_dtype; }
    bool uses_cuda_graph() const { return graph_exec != nullptr; }

    // infer() + copy the labels out as int32 (uint8 / int64 outputs are converted)
    vector<int32_t> inference();

    // with_transfer: time H2D + engine + D2H, i.e. what a deployment actually pays
    void test_speed_fps(bool with_transfer = false);

    vector<int> get_input_shape();
    vector<int> get_output_shape();
};


#endif
