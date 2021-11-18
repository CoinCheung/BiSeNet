
#include <random>
#include <iostream>
#include <string>
#include <memory>
#include <vector>
#include <chrono>
#include <numeric>
#include <functional>

#include "grpc_client.h"
#include "common.h"

#include <opencv2/opencv.hpp>


namespace tc = triton::client;


#define FAIL_IF_ERR(X, MSG)                                        \
  {                                                                \
    tc::Error err = (X);                                          \
    if (!err.IsOk()) {                                             \
      std::cerr << "error: " << (MSG) << ": " << err << std::endl; \
      exit(1);                                                     \
    }                                                              \
  }


// std::string url("10.128.61.8:8001");
std::string url("127.0.0.1:8001");
std::string model_name("bisenetv1");
std::string model_version("1");
uint32_t client_timeout{5000000}; 
bool verbose = false;

std::string impth("../../../example.png"); 
std::string savepth("./res.jpg"); 
std::vector<int64_t> inp_shape{1, 3, 1024, 2048};
std::vector<int64_t> outp_shape{1, 1024, 2048};
std::string inp_name("input_image");
std::string outp_name("preds");
std::string inp_type("FP32");



std::vector<std::vector<uint8_t>> get_color_map();
std::vector<float> get_image(std::string, std::vector<int64_t>&);
void save_predict(std::string, int64_t*, 
        std::vector<int64_t>, std::vector<int64_t>);
void do_inference();
void print_infos();
void test_speed();


int main() {
    // print_infos();
    do_inference();
    // test_speed();
    return 0;
}


void do_inference() {

    // define client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client");
    std::cout << "create client\n";

    // create input 
    std::vector<float> input_data = get_image(impth, inp_shape);
    std::cout << "read image: " << impth << std::endl;

    tc::InferInput* input;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input, inp_name, inp_shape, inp_type), 
        "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);
    FAIL_IF_ERR(input_ptr->Reset(), // reset input
        "unable to reset input data");
    FAIL_IF_ERR(
        input_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input_data[0]),
            input_data.size() * sizeof(float)), // NOTE: float can be others according to input type
        "unable to set data for input");
    std::cout << "set input\n";


    // create output
    tc::InferRequestedOutput* output;
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, outp_name),
        "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);
    std::cout << "set output\n";

    // infer options
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    std::cout << "set options\n";

    // inference
    std::vector<tc::InferInput*> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};
    tc::InferResult* results;
    FAIL_IF_ERR(
        client->Infer(
            &results, options, inputs, outputs, http_headers,
            compression_algorithm),
        "failed sending synchronous infer request");
    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);
    FAIL_IF_ERR(
        results_ptr->RequestStatus(), 
        "inference failed");
    std::cout << "send request and do inference\n";

    // parse output
    int64_t* raw_oup{nullptr}; // NOTE: int64_t is used according to model
    size_t n_bytes{0};
    FAIL_IF_ERR(
        results_ptr->RawData(
            outp_name, (const uint8_t**)(&raw_oup), &n_bytes),
        "fetch output failed");
    if (n_bytes != outp_shape[1] * outp_shape[2] * sizeof(int64_t)) {
        std::cerr << "output shape is not set correctly\n";
        exit(1);
    }
    std::cout << "fetch output\n";

    // save colorful result
    save_predict(savepth, raw_oup, inp_shape, outp_shape);
    std::cout << "save inference result to：" << savepth << std::endl;
}


std::vector<float> get_image(std::string impth, std::vector<int64_t>& shape) {
    int64_t iH = shape[2];
    int64_t iW = shape[3];
    cv::Mat im = cv::imread(impth);
    if (im.empty()) {
        std::cerr << "cv::imread failed: " << impth << std::endl;
        exit(1);
    }
    int64_t orgH{im.rows}, orgW{im.cols};
    if ((orgH != iH) || orgW != iW) {
        std::cout << "resize orignal image of (" << orgH << "," << orgW 
            << ") to (" << iH << ", " << iW << ") according to model requirement\n";
        cv::resize(im, im, cv::Size(iW, iH), cv::INTER_CUBIC);
    }

    std::vector<float> data(iH * iW * 3);
    float mean[3] = {0.3257f, 0.3690f, 0.3223f};
    float var[3] = {0.2112f, 0.2148f, 0.2115f};
    float scale = 1.f / 255.f;
    for (float &el : var) el = 1.f / el;
    for (int h{0}; h < iH; ++h) {
        cv::Vec3b *p = im.ptr<cv::Vec3b>(h);
        for (int w{0}; w < iW; ++w) {
            for (int c{0}; c < 3; ++c) {
                int idx = (2 - c) * iH * iW + h * iW + w; // to rgb order
                data[idx] = (p[w][c] * scale - mean[c]) * var[c];
            }
        }
    }
    return data;
}


std::vector<std::vector<uint8_t>> get_color_map() {
    std::vector<std::vector<uint8_t>> color_map(256, 
            std::vector<uint8_t>(3));
    std::minstd_rand rand_eng(123);
    std::uniform_int_distribution<uint8_t> u(0, 255);
    for (int i{0}; i < 256; ++i) {
        for (int j{0}; j < 3; ++j) {
            color_map[i][j] = u(rand_eng);
        }
    }
    return color_map;
}


void save_predict(std::string savename, int64_t* data, 
        std::vector<int64_t> insize, 
        std::vector<int64_t> outsize) {

    std::vector<std::vector<uint8_t>> color_map = get_color_map();
    int64_t oH = outsize[1];
    int64_t oW = outsize[2];
    cv::Mat pred(cv::Size(oW, oH), CV_8UC3);
    int idx{0};
    for (int i{0}; i < oH; ++i) {
        uint8_t *ptr = pred.ptr<uint8_t>(i);
        for (int j{0}; j < oW; ++j) {
            ptr[0] = color_map[data[idx]][0];
            ptr[1] = color_map[data[idx]][1];
            ptr[2] = color_map[data[idx]][2];
            ptr += 3;
            ++idx;
        }
    }
    cv::imwrite(savename, pred);
}


void print_infos() {
    // define client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client");

    tc::Headers http_headers;
    inference::ModelConfigResponse model_config;
    FAIL_IF_ERR(
        client->ModelConfig(
        &model_config, model_name, model_version, http_headers),
        "unable to get config");

    inference::ModelMetadataResponse model_metadata;
    FAIL_IF_ERR(
        client->ModelMetadata(
        &model_metadata, model_name, model_version, http_headers),
        "unable to get meta data");

    std::cout << "---- model info ----" << std::endl;
    auto input = model_metadata.inputs(0);
    auto output = model_metadata.outputs(0);
    std::cout << "name: " << model_metadata.name() << std::endl;
    std::cout << "platform: " << model_metadata.platform() << std::endl;
    std::cout << "max_batch_size: " << model_config.config().max_batch_size() << std::endl;

    int size;
    size = input.shape().size();
    std::cout << input.name() << ": \n    size: ("; 
    for (int i{0}; i < size; ++i) {
        std::cout << input.shape()[i] << ", "; 
    }
    std::cout << ")\n    data_type: " << input.datatype() << std::endl;;
    size = output.shape().size();
    std::cout << output.name() << ": \n    size: ("; 
    for (int i{0}; i < size; ++i) {
        std::cout << output.shape()[i] << ", "; 
    }
    std::cout << ")\n    data_type: " << output.datatype() << std::endl;;
    std::cout << "--------------------" << std::endl;
}


void test_speed() {
    // define client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, verbose),
        "unable to create grpc client");

    // create input 
    std::vector<float> input_data(std::accumulate(
        inp_shape.begin(), inp_shape.end(), 
        1, std::multiplies<int64_t>())
    );
    tc::InferInput* input;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input, inp_name, inp_shape, inp_type), 
        "unable to get input");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);
    FAIL_IF_ERR(input_ptr->Reset(), // reset input
        "unable to reset input data");
    FAIL_IF_ERR(
        input_ptr->AppendRaw(
            reinterpret_cast<uint8_t*>(&input_data[0]),
            input_data.size() * sizeof(float)), // NOTE: float can be others according to input type
        "unable to set data for input");

    // create output
    tc::InferRequestedOutput* output;
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, outp_name),
        "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);

    // infer options
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;

    // inference
    std::vector<tc::InferInput*> inputs = {input_ptr.get()};
    std::vector<const tc::InferRequestedOutput*> outputs = {output_ptr.get()};
    tc::InferResult* results;

    std::cout << "test speed ... \n";
    const int n_loops{500};
    auto start = std::chrono::steady_clock::now();
    for (int i{0}; i < n_loops; ++i) {
        FAIL_IF_ERR(
            client->Infer(
                &results, options, inputs, outputs, http_headers,
                compression_algorithm),
            "failed sending synchronous infer request");
    }
    auto end = std::chrono::steady_clock::now();

    std::shared_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);
    FAIL_IF_ERR(
        results_ptr->RequestStatus(), 
        "inference failed");

    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    std::cout << "running " << n_loops << " times, use time: "
        << duration << "s" << std::endl; 
    std::cout << "fps is: " << static_cast<double>(n_loops) / duration << std::endl;
}
