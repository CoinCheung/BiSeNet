// #include "NvInfer.h"
// #include "NvOnnxParser.h"
// #include "NvInferPlugin.h"
// #include "NvInferRuntimeCommon.h"

#include <opencv2/opencv.hpp>

#include <iostream>
#include <iterator>
#include <string>
#include <fstream>
#include <vector>
#include <array>
#include <sstream>
#include <random>
#include <cstdint>
#include <unordered_map>

#include "trt_dep.hpp"
#include "read_img.hpp"



using std::string;
using std::ios;
using std::ofstream;
using std::ifstream;
using std::vector;
using std::cout;
using std::endl;
using std::array;
using std::stringstream;

using cv::Mat;




vector<vector<uint8_t>> get_color_map();

void compile_onnx(vector<string> args);
void run_with_trt(vector<string> args);
void test_speed(vector<string> args);


int main(int argc, char* argv[]) {
    CHECK (argc >= 3, "usage is ./segment compile/run/test");

    vector<string> args;
    for (int i{1}; i < argc; ++i) args.emplace_back(argv[i]);

    if (args[0] == "compile") {
        stringstream ss;
        ss << "usage is: ./segment compile input.onnx output.trt [opt_batch_size]\n"
            << "strongly typed: the precision comes from the onnx\n";
        CHECK (argc >= 4, ss.str());
        compile_onnx(args);
    } else if (args[0] == "run") {
        CHECK (argc >= 5, "usage is ./segment run ./xxx.trt input.jpg result.jpg");
        run_with_trt(args);
    } else if (args[0] == "test") {
        CHECK (argc >= 3, "usage is ./segment test ./xxx.trt");
        test_speed(args);
    } else {
        CHECK (false, "usage is ./segment compile/run/test");
    }

    return 0;
}


void compile_onnx(vector<string> args) {
    CHECK (args.size() < 4 || args[3].rfind("--", 0) != 0,
        "precision flags were removed: the precision comes from the onnx");
    int opt_bsize = args.size() > 3 ? std::stoi(args[3]) : 1;

    SemanticSegmentTrt ss_trt;
    ss_trt.set_opt_batch_size(opt_bsize);
    ss_trt.parse_to_engine(args[1]);
    ss_trt.serialize(args[2]);
}


void run_with_trt(vector<string> args) {

    SemanticSegmentTrt ss_trt;
    ss_trt.deserialize(args[1]);

    vector<int> i_dims = ss_trt.get_input_shape();
    vector<int> o_dims = ss_trt.get_output_shape();

    // the network takes NHWC uint8 now, so H and W are dims 1 and 2
    const int iH{i_dims[1]}, iW{i_dims[2]};
    const int oH{o_dims[1]}, oW{o_dims[2]};

    // read straight into the engine's pinned host buffer, no extra copy
    ss_trt.setup_context(1);
    int orgH, orgW;
    read_data(args[2], ss_trt.input_buffer(), iH, iW, orgH, orgW);

    // call engine
    vector<int32_t> res = ss_trt.inference();

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
        cv::resize(pred, pred, cv::Size(orgW, orgH), 0, 0, cv::INTER_LINEAR);
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
    int opt_bsize = 1;
    if (args.size() > 2) opt_bsize = std::stoi(args[2]);

    SemanticSegmentTrt ss_trt;
    ss_trt.set_opt_batch_size(opt_bsize);
    ss_trt.deserialize(args[1]);
    ss_trt.test_speed_fps(false);   // kernel only, comparable with earlier numbers
    ss_trt.test_speed_fps(true);    // what a deployment actually pays
}
