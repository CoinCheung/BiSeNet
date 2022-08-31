
#include <random>
#include <iostream>
#include <fstream>
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


std::string url("10.128.61.8:8001");
// std::string url("127.0.0.1:8001");
std::string model_name("bisenetv1");
std::string model_version("1");
uint32_t client_timeout{5000000}; 

std::string impth("../../../example.png"); 
std::string savepth("./res.jpg"); 
std::vector<float> mean{0.3257, 0.3690, 0.3223}; // city, rgb
std::vector<float> var{0.2112, 0.2148, 0.2115};
std::string inp_name("raw_img_bytes");
std::string outp_name("preds");
std::string inp_type("UINT8");



std::vector<std::vector<uint8_t>> get_color_map();
std::vector<float> get_image(std::string, std::vector<int64_t>&);
std::vector<uint8_t> get_image_bytes(std::string);
void save_predict(std::string, int64_t*, std::vector<int64_t>);
void do_inference();
void do_inference_with_bytes(std::vector<uint8_t>&, bool);
void print_infos();
void test_speed();


int main() {
    // print_infos();
    do_inference();
    // test_speed();
    return 0;
}


void do_inference() {
    // create input 
    // std::vector<float> inp_data = get_image(impth, inp_shape);
    std::vector<uint8_t> inp_data = get_image_bytes(impth);
    std::cout << "read image: " << impth << std::endl;
    do_inference_with_bytes(inp_data, true);
}


void do_inference_with_bytes(std::vector<uint8_t>& inp_data, bool verbose) {

    // define client
    std::unique_ptr<tc::InferenceServerGrpcClient> client;
    FAIL_IF_ERR(
        tc::InferenceServerGrpcClient::Create(&client, url, false), // verbose=false
        "unable to create grpc client");
    if (verbose) std::cout << "create client\n";

    //// raw image
    tc::InferInput* input;
    FAIL_IF_ERR(
        tc::InferInput::Create(&input, inp_name, 
            {1, static_cast<int>(inp_data.size())}, inp_type), 
        "unable to get input data");
    std::shared_ptr<tc::InferInput> input_ptr;
    input_ptr.reset(input);
    FAIL_IF_ERR(input_ptr->Reset(), "unable to reset input data");
    FAIL_IF_ERR(input_ptr->AppendRaw(inp_data), "unable to set data for input");
    //// mean/std
    tc::InferInput *inp_mean, *inp_std;
    FAIL_IF_ERR(
        tc::InferInput::Create(&inp_mean, "channel_mean", {1, 3}, "FP32"), 
        "unable to get input mean");
    FAIL_IF_ERR(
        tc::InferInput::Create(&inp_std, "channel_std", {1, 3}, "FP32"), 
        "unable to get input std");
    std::shared_ptr<tc::InferInput> inp_mean_ptr, inp_std_ptr;
    inp_mean_ptr.reset(inp_mean);
    inp_std_ptr.reset(inp_std);
    FAIL_IF_ERR(inp_mean_ptr->Reset(), "unable to reset input mean");
    FAIL_IF_ERR(inp_std_ptr->Reset(), "unable to reset input std");
    FAIL_IF_ERR(
        inp_mean_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&mean[0]), // must be uint8_t data type
            mean.size() * sizeof(float)),
        "unable to set data for input mean");
    FAIL_IF_ERR(
        inp_std_ptr->AppendRaw(reinterpret_cast<uint8_t*>(&var[0]), 
            var.size() * sizeof(float)),
        "unable to set data for input std");
    if (verbose) std::cout << "set input\n";


    // create output
    tc::InferRequestedOutput* output;
    FAIL_IF_ERR(
        tc::InferRequestedOutput::Create(&output, outp_name),
        "unable to get output");
    std::shared_ptr<tc::InferRequestedOutput> output_ptr;
    output_ptr.reset(output);
    if (verbose) std::cout << "set output\n";

    // infer options
    tc::InferOptions options(model_name);
    options.model_version_ = model_version;
    options.client_timeout_ = client_timeout;
    tc::Headers http_headers;
    grpc_compression_algorithm compression_algorithm =
        grpc_compression_algorithm::GRPC_COMPRESS_NONE;
    if (verbose) std::cout << "set options\n";

    // inference
    std::vector<tc::InferInput*> inputs = {input_ptr.get(), 
        inp_mean_ptr.get(), inp_std_ptr.get()};
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
    if (verbose) std::cout << "send request and do inference\n";

    // parse output
    int64_t* raw_outp{nullptr}; // NOTE: int64_t is used according to model
    size_t n_bytes{0};
    FAIL_IF_ERR(
        results_ptr->RawData(
            outp_name, (const uint8_t**)(&raw_outp), &n_bytes),
        "fetch output failed");
    std::vector<int64_t> outp_shape;
    FAIL_IF_ERR(
        results_ptr->Shape(outp_name, &outp_shape),
        "get output shape failed");
    if (n_bytes != std::accumulate(outp_shape.begin(), outp_shape.end(), 1,
                std::multiplies<int64_t>()) * sizeof(int64_t)) {
        std::cerr << "output shape is not set correctly\n";
        exit(1);
    }
    if (verbose) std::cout << "fetch output\n";

    // save colorful result
    save_predict(savepth, raw_outp, outp_shape);
    if (verbose) std::cout << "save inference result to：" << savepth << std::endl;
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


std::vector<uint8_t> get_image_bytes(std::string impth) {
    std::ifstream fin(impth, std::ios::in|std::ios::binary);
    fin.seekg(0, fin.end);
    int nbytes = fin.tellg();
    if (nbytes == -1) {
        std::cerr << "image file read failed: " << impth << std::endl;
        exit(1);
    }
    fin.clear();
    fin.seekg(0);

    std::vector<uint8_t> res(nbytes);
    fin.read(reinterpret_cast<char*>(&res[0]), nbytes);
    fin.close();

    return res;
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
        std::vector<int64_t> outsize) {
    std::vector<std::vector<uint8_t>> color_map = get_color_map();
    int64_t oH = outsize[2]; // outsize is n1hw
    int64_t oW = outsize[3];
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
        tc::InferenceServerGrpcClient::Create(&client, url, false),
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

    std::vector<uint8_t> inp_data = get_image_bytes(impth);
    // warmup
    do_inference_with_bytes(inp_data, false);

    std::cout << "test speed ... \n";
    const int n_loops{500};
    auto start = std::chrono::steady_clock::now();
    for (int i{0}; i < n_loops; ++i) {
        do_inference_with_bytes(inp_data, false);
    }
    auto end = std::chrono::steady_clock::now();

    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    std::cout << "running " << n_loops << " times, use time: "
        << duration << "s" << std::endl;
    std::cout << "fps is: " << static_cast<double>(n_loops) / duration << std::endl;
}
