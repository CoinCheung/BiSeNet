
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <unordered_map>
#include <sstream>
#include <chrono>
#include <iterator>

#include "trt_dep.hpp"
#include "argmax_plugin.h"
#include "batch_stream.hpp"
#include "entropy_calibrator.hpp"


using nvinfer1::IHostMemory;
using nvinfer1::IBuilder;
using nvinfer1::INetworkDefinition;
using nvinfer1::ICudaEngine;
using nvinfer1::IInt8Calibrator;
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



void SemanticSegmentTrt::parse_to_engine(string onnx_pth, 
        string quant, string data_root, string data_file) {

    std::unique_ptr<ArgMaxPluginCreator> plugin_creator{new ArgMaxPluginCreator{}};
    plugin_creator->setPluginNamespace("");
    bool status = getPluginRegistry()->registerCreator(*plugin_creator.get(), "");
    CHECK(status, "failed to register plugin");

    auto builder = TrtUnqPtr<IBuilder>(nvinfer1::createInferBuilder(gLogger));
    CHECK(static_cast<bool>(builder), "create builder failed");

    auto network = TrtUnqPtr<INetworkDefinition>(builder->createNetworkV2(0));
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
    Dims in_dims = network->getInput(0)->getDimensions();
    int32_t C = in_dims.d[1], H = in_dims.d[2], W = in_dims.d[3];
    Dims dmin = Dims4{1, C, H, W};
    Dims dopt = Dims4{opt_bsize, C, H, W};
    Dims dmax = Dims4{32, C, H, W};
    profile->setDimensions(input->getName(), OptProfileSelector::kMIN, dmin);
    profile->setDimensions(input->getName(), OptProfileSelector::kOPT, dopt);
    profile->setDimensions(input->getName(), OptProfileSelector::kMAX, dmax);
    config->addOptimizationProfile(profile);

    config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 1UL << 32);

    if (quant == "fp16" or quant == "int8") { // fp16
        if (builder->platformHasFastFp16() == false) {
            cout << "fp16 is set, but platform does not support, so we ignore this\n";
        } else {
            config->setFlag(nvinfer1::BuilderFlag::kFP16); 
        }
    }
    if (quant == "bf16") { // bf16
        config->setFlag(nvinfer1::BuilderFlag::kBF16); 
    }
    if (quant == "fp8") { // fp8
        config->setFlag(nvinfer1::BuilderFlag::kFP8); 
    }

    std::unique_ptr<IInt8Calibrator> calibrator;
    if (quant == "int8") { // int8
        if (builder->platformHasFastInt8() == false) {
            cout << "int8 is set, but platform does not support, so we ignore this\n";
        } else {

            int batchsize = 32;
            int n_cal_batches = -1;
            string cal_table_name = "calibrate_int8";

            Dims indim = network->getInput(0)->getDimensions();
            BatchStream calibrationStream(
                    batchsize, n_cal_batches, indim,
                    data_root, data_file);

            config->setFlag(nvinfer1::BuilderFlag::kINT8); 

            calibrator.reset(new Int8EntropyCalibrator2<BatchStream>(
                calibrationStream, 0, cal_table_name.c_str(), input_name.c_str(), false));
            config->setInt8Calibrator(calibrator.get());
        }
    }

    // output->setType(nvinfer1::DataType::kINT32);
    // output->setType(nvinfer1::DataType::kFLOAT);

    cout << "start to build \n";

    auto plan = TrtUnqPtr<IHostMemory>(builder->buildSerializedNetwork(*network, *config));
    CHECK(static_cast<bool>(plan), "build serialized engine failed");

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    CHECK(static_cast<bool>(runtime), "create runtime failed");

    engine.reset(runtime->deserializeCudaEngine(plan->data(), plan->size()));
    CHECK(static_cast<bool>(engine), "deserialize engine failed");
    cout << "done build engine \n";
}


void SemanticSegmentTrt::set_opt_batch_size(int bs) {
    CHECK(bs > 0 and bs < 33, "batch size should be less than 32");
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

    std::unique_ptr<ArgMaxPluginCreator> plugin_creator{new ArgMaxPluginCreator{}};
    plugin_creator->setPluginNamespace("");
    bool status = getPluginRegistry()->registerCreator(*plugin_creator.get(), "");
    CHECK(status, "failed to register plugin");

    runtime.reset(nvinfer1::createInferRuntime(gLogger));
    engine.reset(runtime->deserializeCudaEngine((void*)&buf[0], mdsize));

    input_name = engine->getIOTensorName(0);
    output_name = engine->getIOTensorName(1);
}


vector<int64_t> SemanticSegmentTrt::inference(vector<float>& data) {
    Dims in_dims = engine->getTensorShape(input_name.c_str());
    Dims out_dims = engine->getTensorShape(output_name.c_str());

    const int64_t batchsize{1}, H{out_dims.d[1]}, W{out_dims.d[2]};
    const int64_t in_size{static_cast<int64_t>(data.size())};
    const int64_t out_size{batchsize * H * W};

    Dims4 in_shape(batchsize, in_dims.d[1], in_dims.d[2], in_dims.d[3]);

    vector<void*> buffs(2, nullptr);
    vector<int64_t> res(out_size);

    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    CHECK(state == cudaSuccess, "allocate memory failed");
    state = cudaMalloc(&buffs[1], out_size * sizeof(int64_t));
    CHECK(state == cudaSuccess, "allocate memory failed");

    state = cudaMemcpyAsync(
            buffs[0], &data[0], in_size * sizeof(float),
            cudaMemcpyHostToDevice, *stream);
    CHECK(state == cudaSuccess, "transmit to device failed");

    auto context = TrtUnqPtr<IExecutionContext>(engine->createExecutionContext());
    CHECK(static_cast<bool>(context), "create execution context failed");

    // Dynamic shape require this setInputShape
    bool success = context->setInputShape(input_name.c_str(), in_shape);
    CHECK(success, "set input shape failed");
    context->setInputTensorAddress(input_name.c_str(), buffs[0]);
    context->setOutputTensorAddress(output_name.c_str(), buffs[1]);
    context->enqueueV3(*stream);

    state = cudaMemcpyAsync(
            &res[0], buffs[1], out_size * sizeof(int64_t),
            cudaMemcpyDeviceToHost, *stream);
    CHECK(state == cudaSuccess, "transmit back to host failed");

    cudaStreamSynchronize(*stream);

    for (auto buf : buffs) {
        cudaFree(buf);
    }

    return res;
}


void SemanticSegmentTrt::test_speed_fps() {
    Dims in_dims = engine->getTensorShape(input_name.c_str());
    Dims out_dims = engine->getTensorShape(output_name.c_str());

    const int64_t batchsize{opt_bsize};
    const int64_t oH{out_dims.d[1]}, oW{out_dims.d[2]};
    const int64_t iH{in_dims.d[2]}, iW{in_dims.d[3]};
    const int64_t in_size{batchsize * 3 * iH * iW};
    const int64_t out_size{batchsize * oH * oW};

    Dims4 in_shape(batchsize, in_dims.d[1], in_dims.d[2], in_dims.d[3]);

    vector<void*> buffs(2, nullptr);
    cudaError_t state;
    state = cudaMalloc(&buffs[0], in_size * sizeof(float));
    CHECK(state == cudaSuccess, "allocate memory failed");
    state = cudaMalloc(&buffs[1], out_size * sizeof(int64_t));
    CHECK(state == cudaSuccess, "allocate memory failed");

    auto context = TrtUnqPtr<IExecutionContext>(engine->createExecutionContext());
    CHECK(static_cast<bool>(context), "create execution context failed");
    bool success = context->setInputShape(input_name.c_str(), in_shape);
    CHECK(success, "set input shape failed");

    cout << "\ntest with cropsize of (" << iH << ", " << iW << "), "
        << "and batch size of " << batchsize << " ...\n";
    context->executeV2(buffs.data()); // run one batch ahead
    auto start = std::chrono::steady_clock::now();
    const int n_loops{2000};
    for (int i{0}; i < n_loops; ++i) {
        context->executeV2(buffs.data());
    }
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    int n_frames = n_loops * batchsize;
    cout << "running " << n_loops << " times, use time: "
        << duration << "s" << endl; 
    cout << "fps is: " << static_cast<double>(n_frames) / duration << endl;

    for (auto buf : buffs) {
        cudaFree(buf);
    }
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
