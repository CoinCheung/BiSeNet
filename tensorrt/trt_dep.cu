
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <iterator>
#include <cstdlib>

#include "trt_dep.hpp"


using nvinfer1::IHostMemory;
using nvinfer1::IBuilder;
using nvinfer1::INetworkDefinition;
using nvinfer1::ICudaEngine;
using nvinfer1::IBuilderConfig;
using nvinfer1::IRuntime;
using nvinfer1::IExecutionContext;
using nvinfer1::ILogger;
using nvinfer1::Dims;
using nvinfer1::Dims4;
using nvinfer1::OptProfileSelector;
using Severity = nvinfer1::ILogger::Severity;

using std::string;
using std::ios;
using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;
using std::array;


Logger gLogger;



void CHECK(bool condition, string msg) {
    if (!condition) {
        cout << msg << endl;;
        std::terminate();
    }
}


void CHECK_CUDA(cudaError_t state, string msg) {
    if (state != cudaSuccess) {
        cout << msg << ": " << cudaGetErrorString(state) << endl;
        std::terminate();
    }
}


void SemanticSegmentTrt::parse_to_engine(string onnx_pth) {

    auto builder = TrtUnqPtr<IBuilder>(nvinfer1::createInferBuilder(gLogger));
    CHECK(static_cast<bool>(builder), "create builder failed");

    // strongly typed: precision from the onnx (types, Q/DQ)
    uint32_t network_flags = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
    auto network = TrtUnqPtr<INetworkDefinition>(builder->createNetworkV2(network_flags));
    CHECK(static_cast<bool>(network), "create network failed");

    auto parser = TrtUnqPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    CHECK(static_cast<bool>(parser), "create parser failed");

    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    bool success = parser->parseFromFile(onnx_pth.c_str(), verbosity);
    CHECK(success, "parse onnx file failed");

    if (network->getNbInputs() != 1) {
        cout << "expect model to have only one input, but this model has " 
            << network->getNbInputs() << endl;
        std::terminate();
    }
    auto input = network->getInput(0);
    auto output = network->getOutput(0);
    input_name = input->getName();
    output_name = output->getName();

    auto config = TrtUnqPtr<IBuilderConfig>(builder->createBuilderConfig());
    CHECK(static_cast<bool>(config), "create builder config failed");

    config->setProfileStream(*stream);

    auto profile = builder->createOptimizationProfile();
    // only the batch dim varies; the others stay as the onnx declares them
    Dims in_dims = network->getInput(0)->getDimensions();
    int32_t d1 = in_dims.d[1], d2 = in_dims.d[2], d3 = in_dims.d[3];
    Dims dmin = Dims4{1, d1, d2, d3};
    Dims dopt = Dims4{opt_bsize, d1, d2, d3};
    Dims dmax = Dims4{kMaxBatch, d1, d2, d3};
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, dmin);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, dopt);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, dmax);
    config->addOptimizationProfile(profile);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 34); // 16G
    config->setBuilderOptimizationLevel(5);


    // output->setType(nvinfer1::DataType::kINT32);
    // output->setType(nvinfer1::DataType::kFLOAT);

    cout << "start to build \n";

    auto plan = TrtUnqPtr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    CHECK(static_cast<bool>(plan), "build serialized engine failed");

    // a previous engine: context, then engine, then its runtime
    release_context();
    engine.reset();
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    CHECK(static_cast<bool>(runtime), "create runtime failed");

    engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    CHECK(static_cast<bool>(engine), "deserialize engine failed");
    cout << "done build engine \n";
}


void SemanticSegmentTrt::set_opt_batch_size(int bs) {
    CHECK(bs > 0 and bs <= kMaxBatch,
          "batch size must be in [1, " + std::to_string(kMaxBatch)
          + "], got " + std::to_string(bs)
          + " (this is the profile range built by parse_to_engine)");
    opt_bsize = bs;
}


void SemanticSegmentTrt::serialize(string save_path) {

    auto trt_stream = TrtUnqPtr<IHostMemory>(engine->serialize());
    CHECK(static_cast<bool>(trt_stream), "serialize engine failed");

    ofstream ofile(save_path, ios::out | ios::binary);
    ofile.write((const char*)trt_stream->data(), trt_stream->size());

    ofile.close();
}


void SemanticSegmentTrt::deserialize(string serpth) {

    ifstream ifile(serpth, ios::in | ios::binary);
    CHECK(static_cast<bool>(ifile), "read serialized file failed");

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    release_context();   // a previous engine: context, then engine, then its runtime
    engine.reset();
    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine((void*)&buf[0], mdsize));
    // e.g. a plugin missing from this binary, or a TensorRT version mismatch
    CHECK(static_cast<bool>(engine), "deserialize engine failed (see the TensorRT error above)");

    // pick the tensors by I/O mode, not by index
    input_name.clear();
    output_name.clear();
    for (int i{0}; i < engine->getNbIOTensors(); ++i) {
        const char* n = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(n) == nvinfer1::TensorIOMode::kINPUT) {
            CHECK(input_name.empty(), "engine has more than one input");
            input_name = n;
        } else {
            CHECK(output_name.empty(), "engine has more than one output");
            output_name = n;
        }
    }
    CHECK(!input_name.empty() && !output_name.empty(), "engine needs one input and one output");
}


void SemanticSegmentTrt::release_context() {
    if (graph_exec != nullptr) {
        cudaGraphExecDestroy(graph_exec);
        graph_exec = nullptr;
    }
    context.reset();
    if (d_input  != nullptr) { cudaFree(d_input);  d_input  = nullptr; }
    if (d_output != nullptr) { cudaFree(d_output); d_output = nullptr; }
    // pinned buffers come from cudaHostAlloc, TRT_NO_PINNED=1 ones from malloc
    for (void** h : {&h_input, &h_output}) {
        if (*h == nullptr) continue;
        if (use_pinned) cudaFreeHost(*h); else free(*h);
        *h = nullptr;
    }
    in_numel = out_numel = 0;
    in_elem_size = out_elem_size = 0;
    ctx_bsize = 0;
}


void SemanticSegmentTrt::setup_context(int bs) {
    if (ctx_bsize == bs) return; // already bound to this batch size
    release_context();

    Dims in_dims = engine->getTensorShape(input_name.c_str());
    Dims out_dims = engine->getTensorShape(output_name.c_str());
    // images are resized to the engine's H x W, so only the batch may be dynamic
    CHECK(in_dims.nbDims == 4 && in_dims.d[1] > 0 && in_dims.d[2] > 0 && in_dims.d[3] > 0
          && out_dims.nbDims == 3 && out_dims.d[1] > 0 && out_dims.d[2] > 0,
          "only the batch dimension may be dynamic (input HxWxC and output HxW must be static)");

    if (engine->getNbOptimizationProfiles() > 0) {
        Dims pmin = engine->getProfileShape(input_name.c_str(), 0, OptProfileSelector::kMIN);
        Dims pmax = engine->getProfileShape(input_name.c_str(), 0, OptProfileSelector::kMAX);
        CHECK(bs >= pmin.d[0] and bs <= pmax.d[0],
              "batch size " + std::to_string(bs) + " is outside this engine's profile ["
              + std::to_string(pmin.d[0]) + ", " + std::to_string(pmax.d[0])
              + "]; rebuild the engine with a wider profile or use a batch size in range");
    }

    in_numel = static_cast<int64_t>(bs) * in_dims.d[1] * in_dims.d[2] * in_dims.d[3];
    // the graph now takes raw uint8 pixels, so never assume float here
    switch (engine->getTensorDataType(input_name.c_str())) {
        case nvinfer1::DataType::kUINT8:
        case nvinfer1::DataType::kINT8:  in_elem_size = 1; break;
        case nvinfer1::DataType::kHALF:  in_elem_size = 2; break;
        case nvinfer1::DataType::kFLOAT: in_elem_size = 4; break;
        default: CHECK(false, "unsupported input dtype"); break;
    }
    out_numel = static_cast<int64_t>(bs) * out_dims.d[1] * out_dims.d[2];
    // uint8 labels (a Cast after the argmax) cut the D2H copy to a quarter
    out_dtype = engine->getTensorDataType(output_name.c_str());
    switch (out_dtype) {
        case nvinfer1::DataType::kINT32: out_elem_size = 4; break;
        case nvinfer1::DataType::kUINT8: out_elem_size = 1; break;
        case nvinfer1::DataType::kINT64: out_elem_size = 8; break;
        default: CHECK(false, "unsupported output dtype, expected int32, uint8 or int64"); break;
    }

    CHECK_CUDA(cudaMalloc(&d_input, in_numel * in_elem_size), "allocate input failed");
    CHECK_CUDA(cudaMalloc(&d_output, out_numel * out_elem_size), "allocate output failed");

    // pinned host buffers (pageable halves H2D); TRT_NO_PINNED=1 uses malloc
    const char* no_pinned = std::getenv("TRT_NO_PINNED");
    use_pinned = !(no_pinned != nullptr && string(no_pinned) == "1");
    if (use_pinned) {
        CHECK_CUDA(cudaHostAlloc(&h_input, in_numel * in_elem_size, cudaHostAllocDefault),
                   "allocate pinned input failed");
        CHECK_CUDA(cudaHostAlloc(&h_output, out_numel * out_elem_size,
                   cudaHostAllocDefault), "allocate pinned output failed");
    } else {
        h_input = malloc(in_numel * in_elem_size);
        h_output = malloc(out_numel * out_elem_size);
        CHECK(h_input != nullptr && h_output != nullptr, "allocate host buffers failed");
    }
    // zeroed: random bits can be NaN/denormal and skew timing
    CHECK_CUDA(cudaMemset(d_input, 0, in_numel * in_elem_size), "memset input failed");

    context.reset(engine->createExecutionContext());
    CHECK(static_cast<bool>(context), "create execution context failed");

    // Dynamic shape require this setInputShape
    Dims4 in_shape(bs, in_dims.d[1], in_dims.d[2], in_dims.d[3]);
    bool success = context->setInputShape(input_name.c_str(), in_shape);
    CHECK(success, "set input shape failed");
    context->setInputTensorAddress(input_name.c_str(), d_input);
    context->setOutputTensorAddress(output_name.c_str(), d_output);

    // warm up first: some tactics initialize lazily, which cannot be captured
    for (int i{0}; i < 3; ++i) {
        CHECK(context->enqueueV3(*stream), "enqueue failed");
    }
    CHECK_CUDA(cudaStreamSynchronize(*stream), "warm up failed");

    // escape hatch for A/B measurement and for debugging capture problems
    const char* no_graph = std::getenv("TRT_NO_CUDA_GRAPH");
    if (no_graph != nullptr && string(no_graph) == "1") {
        ctx_bsize = bs;
        cout << "TRT_NO_CUDA_GRAPH=1, using plain enqueueV3" << endl;
        return;
    }

    cudaGraph_t graph{nullptr};
    cudaError_t state = cudaStreamBeginCapture(*stream, cudaStreamCaptureModeThreadLocal);
    if (state == cudaSuccess) {
        bool enq = context->enqueueV3(*stream);
        state = cudaStreamEndCapture(*stream, &graph);
        if (enq && state == cudaSuccess) {
            state = cudaGraphInstantiate(&graph_exec, graph, 0);
            if (state != cudaSuccess) graph_exec = nullptr;
        }
    }
    if (graph != nullptr) cudaGraphDestroy(graph);
    ctx_bsize = bs;

    if (graph_exec == nullptr) {
        cout << "WARNING: cuda graph capture failed ("
             << cudaGetErrorString(cudaGetLastError())
             << "), falling back to enqueueV3" << endl;
        // the failed capture may have left an error latched on the stream
        cudaGetLastError();
        CHECK_CUDA(cudaStreamSynchronize(*stream), "stream recovery failed");
    }
}


void SemanticSegmentTrt::launch_once() {
    if (graph_exec != nullptr) {
        CHECK_CUDA(cudaGraphLaunch(graph_exec, *stream), "launch cuda graph failed");
    } else {
        CHECK(context->enqueueV3(*stream), "enqueue failed");
    }
}


void SemanticSegmentTrt::infer() {
    CHECK(ctx_bsize > 0, "call setup_context() and fill input_buffer() first");

    CHECK_CUDA(cudaMemcpyAsync(
            d_input, h_input, in_numel * in_elem_size,
            cudaMemcpyHostToDevice, *stream), "transmit to device failed");

    launch_once();

    CHECK_CUDA(cudaMemcpyAsync(
            h_output, d_output, out_numel * out_elem_size,
            cudaMemcpyDeviceToHost, *stream), "transmit back to host failed");

    CHECK_CUDA(cudaStreamSynchronize(*stream), "inference failed");
}


vector<int32_t> SemanticSegmentTrt::inference() {
    infer();
    if (out_dtype == nvinfer1::DataType::kUINT8) {
        const uint8_t* p = static_cast<const uint8_t*>(h_output);
        return vector<int32_t>(p, p + out_numel);
    }
    if (out_dtype == nvinfer1::DataType::kINT64) {
        const int64_t* p = static_cast<const int64_t*>(h_output);
        return vector<int32_t>(p, p + out_numel);
    }
    const int32_t* p = static_cast<const int32_t*>(h_output);
    return vector<int32_t>(p, p + out_numel);
}


void SemanticSegmentTrt::test_speed_fps(bool with_transfer) {
    Dims in_dims = engine->getTensorShape(input_name.c_str());

    const int64_t batchsize{opt_bsize};
    // the graph is NHWC: H and W are dims 1 and 2
    const int64_t iH{in_dims.d[1]}, iW{in_dims.d[2]};

    setup_context(static_cast<int>(batchsize));

    cout << "\ntest with cropsize of (" << iH << ", " << iW << "), "
        << "and batch size of " << batchsize << " ...\n";
    cout << "host transfers (H2D+D2H): " << (with_transfer ? "included" : "excluded") << endl;
    if (out_dtype != nvinfer1::DataType::kINT32)
        cout << "output dtype: " << (out_dtype == nvinfer1::DataType::kUINT8 ? "uint8" : "int64") << endl;

    // run a few batches ahead so clocks and caches settle
    for (int i{0}; i < 20; ++i) launch_once();
    CHECK_CUDA(cudaStreamSynchronize(*stream), "warm up failed");

    auto start = std::chrono::steady_clock::now();
    const int n_loops{2000};
    for (int i{0}; i < n_loops; ++i) {
        if (with_transfer) {
            CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, in_numel * in_elem_size,
                    cudaMemcpyHostToDevice, *stream), "h2d failed");
        }
        launch_once();
        if (with_transfer) {
            CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, out_numel * out_elem_size,
                    cudaMemcpyDeviceToHost, *stream), "d2h failed");
        }
        // sync per iteration, like the old blocking executeV2
        CHECK_CUDA(cudaStreamSynchronize(*stream), "inference failed");
    }
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    int n_frames = n_loops * batchsize;
    cout << "running " << n_loops << " times, use time: "
        << duration << "s" << endl; 
    cout << "fps is: " << static_cast<double>(n_frames) / duration << endl;
    cout << "latency per batch: " << duration * 1000. / n_loops << " ms" << endl;
    cout << "latency per frame: " << duration * 1000. / n_frames << " ms" << endl;
}


vector<int> SemanticSegmentTrt::get_input_shape() {

    Dims i_dims = engine->getTensorShape(input_name.c_str());
    vector<int> res(i_dims.d, i_dims.d + i_dims.nbDims);
    return res;
}


vector<int> SemanticSegmentTrt::get_output_shape() {

    Dims o_dims = engine->getTensorShape(output_name.c_str());
    vector<int> res(o_dims.d, o_dims.d + o_dims.nbDims);
    return res;
}
