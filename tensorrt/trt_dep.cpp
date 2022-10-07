
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <unordered_map>
#include <sstream>
#include <chrono>

#include "trt_dep.hpp"
#include "batch_stream.hpp"
#include "entropy_calibrator.hpp"
#include "kernels.hpp"


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


Logger gLogger;


TrtSharedEnginePtr shared_engine_ptr(ICudaEngine* ptr) {
    return TrtSharedEnginePtr(ptr, TrtDeleter());
}


TrtSharedEnginePtr parse_to_engine(string onnx_pth, 
        string quant, string data_root, string data_file) {
    unsigned int maxBatchSize{1};
    long memory_limit = 1UL << 32; // 4G

    auto builder = TrtUnqPtr<IBuilder>(nvinfer1::createInferBuilder(gLogger));
    if (!builder) {
        cout << "create builder failed\n";
        std::abort();
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = TrtUnqPtr<INetworkDefinition>(
            builder->createNetworkV2(explicitBatch));
    if (!network) {
        cout << "create network failed\n";
        std::abort();
    }

    auto config = TrtUnqPtr<IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        cout << "create builder config failed\n";
        std::abort();
    }

    auto parser = TrtUnqPtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, gLogger));
    if (!parser) {
        cout << "create parser failed\n";
        std::abort();
    }

    int verbosity = (int)nvinfer1::ILogger::Severity::kWARNING;
    bool state = parser->parseFromFile(onnx_pth.c_str(), verbosity);
    if (!state) {
        cout << "parse model failed\n";
        std::abort();
    }

    config->setMaxWorkspaceSize(memory_limit);
    if ((quant == "fp16" or quant == "int8") && builder->platformHasFastFp16()) {
        config->setFlag(nvinfer1::BuilderFlag::kFP16); // fp16
    }
    std::unique_ptr<IInt8Calibrator> calibrator;
    if (quant == "int8" && builder->platformHasFastInt8()) {
        config->setFlag(nvinfer1::BuilderFlag::kINT8); //int8
        int batchsize = 32;
        int n_cal_batches = -1;
        string cal_table_name = "calibrate_int8";
        string input_name = "input_image";

        Dims indim = network->getInput(0)->getDimensions();
        BatchStream calibrationStream(
                batchsize, n_cal_batches, indim,
                data_root, data_file);
        calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
            calibrationStream, 0, cal_table_name.c_str(), input_name.c_str()));
        config->setInt8Calibrator(calibrator.get());
    }

    auto output = network->getOutput(0);
    // output->setType(nvinfer1::DataType::kINT32);
    output->setType(nvinfer1::DataType::kFLOAT);

    cout << " start to build \n";
    CudaStreamUnqPtr stream(new cudaStream_t);
    if (cudaStreamCreate(stream.get())) {
        cout << "create stream failed\n";
        std::abort();
    }
    config->setProfileStream(*stream);

    auto plan = TrtUnqPtr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    if (!plan) {
        cout << "serialization failed\n";
        std::abort();
    }

    auto runtime = TrtUnqPtr<IRuntime>(nvinfer1::createInferRuntime(gLogger));
    if (!plan) {
        cout << "create runtime failed\n";
        std::abort();
    }

    TrtSharedEnginePtr engine = shared_engine_ptr(
            runtime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!engine) {
        cout << "create engine failed\n";
        std::abort();
    }
    cout << "done build engine \n";

    return engine;
}


void serialize(TrtSharedEnginePtr engine, string save_path) {

    auto trt_stream = TrtUnqPtr<IHostMemory>(engine->serialize());
    if (!trt_stream) {
        cout << "serialize engine failed\n";
        std::abort();
    }

    ofstream ofile(save_path, ios::out | ios::binary);
    ofile.write((const char*)trt_stream->data(), trt_stream->size());

    ofile.close();
}


TrtSharedEnginePtr deserialize(string serpth) {

    ifstream ifile(serpth, ios::in | ios::binary);
    if (!ifile) {
        cout << "read serialized file failed\n";
        std::abort();
    }

    ifile.seekg(0, ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, ios::beg);
    vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();
    cout << "model size: " << mdsize << endl;

    auto runtime = TrtUnqPtr<IRuntime>(nvinfer1::createInferRuntime(gLogger));
    TrtSharedEnginePtr engine = shared_engine_ptr(
            runtime->deserializeCudaEngine((void*)&buf[0], mdsize));
    return engine;
}


vector<int> infer_with_engine(TrtSharedEnginePtr engine, vector<float>& data) {
    Dims3 out_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));

    const int batchsize{1}, H{out_dims.d[2]}, W{out_dims.d[3]};
    const int n_classes{out_dims.d[1]};
    const int in_size{static_cast<int>(data.size())};
    const int logits_size{batchsize * n_classes * H * W};
    const int out_size{batchsize * H * W};
    vector<void*> buffs(3);
    vector<int> res(out_size);

    auto context = TrtUnqPtr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        cout << "create execution context failed\n";
        std::abort();
    }

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[1], logits_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[2], out_size * sizeof(int));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    CudaStreamUnqPtr stream(new cudaStream_t);
    if (cudaStreamCreate(stream.get())) {
        cout << "create stream failed\n";
        std::abort();
    }

    state = cudaMemcpyAsync(
            buffs[0], &data[0], in_size * sizeof(float),
            cudaMemcpyHostToDevice, *stream);
    if (state) {
        cout << "transmit to device failed\n";
        std::abort();
    }

    context->enqueueV2(&buffs[0], *stream, nullptr);
    // context->enqueue(1, &buffs[0], stream, nullptr);
    argMaxFunc(buffs[1], buffs[2], batchsize, n_classes, H * W, stream.get());

    state = cudaMemcpyAsync(
            &res[0], buffs[2], out_size * sizeof(int),
            cudaMemcpyDeviceToHost, *stream);
    if (state) {
        cout << "transmit to host failed \n";
        std::abort();
    }

    cudaStreamSynchronize(*stream);

    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaFree(buffs[2]);

    return res;
}


void test_fps_with_engine(TrtSharedEnginePtr engine) {
    Dims3 in_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("input_image")));
    Dims3 out_dims = static_cast<Dims3&&>(
        engine->getBindingDimensions(engine->getBindingIndex("preds")));

    const int batchsize{1};
    const int oH{out_dims.d[2]}, oW{out_dims.d[3]};
    const int n_classes{out_dims.d[1]};
    const int iH{in_dims.d[2]}, iW{in_dims.d[3]};
    const int in_size{batchsize * 3 * iH * iW};
    const int logits_size{batchsize * n_classes * oH * oW};
    const int out_size{batchsize * oH * oW};

    auto context = TrtUnqPtr<IExecutionContext>(engine->createExecutionContext());
    if (!context) {
        cout << "create execution context failed\n";
        std::abort();
    }

    vector<void*> buffs(3);
    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n"; 
        std::abort();
    }
    state = cudaMalloc(&buffs[1], logits_size * sizeof(float));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }
    state = cudaMalloc(&buffs[2], out_size * sizeof(int));
    if (state) {
        cout << "allocate memory failed\n";
        std::abort();
    }

    cout << "\ntest with cropsize of (" << iH << ", " << iW << ") ...\n";
    auto start = std::chrono::steady_clock::now();
    const int n_loops{1000};
    for (int i{0}; i < n_loops; ++i) {
        // context->execute(1, &buffs[0]);
        context->executeV2(&buffs[0]);
        argMaxFunc(buffs[1], buffs[2], batchsize, n_classes, oH * oW, nullptr);
    }
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    cout << "running " << n_loops << " times, use time: "
        << duration << "s" << endl; 
    cout << "fps is: " << static_cast<double>(n_loops) / duration << endl;


    cudaFree(buffs[0]);
    cudaFree(buffs[1]);
    cudaFree(buffs[2]);
}

