
#include <random>
#include <iostream>
#include <string>
#include <vector>
#include <chrono>

#include <opencv2/opencv.hpp>
#include <inference_engine.hpp>


std::string mdpth("../output_v2/model_v2_city.xml");
std::string device("CPU"); // GNA does not support argmax, my cpu does not has integrated gpu
std::string impth("../../example.png"); 
std::string savepth("./res.jpg");



void get_image(std::string, std::vector<unsigned long>, float*);
std::vector<std::vector<uint8_t>> get_color_map();
void save_predict(std::string, int*, 
        std::vector<unsigned long>, std::vector<unsigned long>);
void print_infos();
void inference();
void test_speed();


int main() {
    // print_infos();
    inference();
    // test_speed();
    return 0;
}


void inference() {

    // model setup
    std::cout << "load network: " << mdpth << std::endl;
    InferenceEngine::Core ie;
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(mdpth);
    model.setBatchSize(1U);

    InferenceEngine::InputsDataMap inputs(model.getInputsInfo());
    InferenceEngine::InputInfo::Ptr input_info = inputs.begin()->second;
    input_info->setPrecision(InferenceEngine::Precision::FP32);
    input_info->setLayout(InferenceEngine::Layout::NCHW);

    InferenceEngine::DataPtr output_info = model.getOutputsInfo().begin()->second;
    output_info->setPrecision(InferenceEngine::Precision::I32);


    InferenceEngine::ExecutableNetwork network = ie.LoadNetwork(model, device);
    InferenceEngine::InferRequest infer_request = network.CreateInferRequest();


    // set input data    
    std::cout << "set input data from: " << impth << std::endl;
    std::string in_name = inputs.begin()->first;
    auto insize = input_info->getTensorDesc().getDims();

    InferenceEngine::Blob::Ptr inblob = infer_request.GetBlob(in_name);
    InferenceEngine::MemoryBlob::Ptr minput = InferenceEngine::as<InferenceEngine::MemoryBlob>(inblob);
    if (!minput) {
        std::cerr << "We expect MemoryBlob from inferRequest, but by fact we "
                             "were not able to cast inputBlob to MemoryBlob"
                          << std::endl;
        std::abort();
    }
    auto minputHolder = minput->wmap();
    float* p_inp = minputHolder.as<float*>();

    get_image(impth, insize, p_inp);


    // inference synchronized 
    std::cout << "do inference " << std::endl;
    infer_request.Infer();


    // fetch output data
    std::cout << "save result to: " << savepth << std::endl;
    std::string out_name = model.getOutputsInfo().begin()->first;
    auto outsize = output_info->getTensorDesc().getDims();
    InferenceEngine::Blob::Ptr outblob = infer_request.GetBlob(out_name);
    InferenceEngine::MemoryBlob::Ptr moutput = InferenceEngine::as<InferenceEngine::MemoryBlob>(outblob);
    auto moutputHolder = moutput->rmap();
    int* p_outp = moutputHolder.as<int*>();

    save_predict(savepth, p_outp, insize, outsize);

}


void get_image(std::string impth, std::vector<unsigned long> insize, float* data) {
    int iH = insize[2];
    int iW = insize[3];
    cv::Mat im = cv::imread(impth);
    if (im.empty()) {
        std::cerr << "cv::imread failed: " << impth << std::endl;
        std::abort();
    }
    int orgH{im.rows}, orgW{im.cols};
    if ((orgH != iH) || orgW != iW) {
        std::cout << "resize orignal image of (" << orgH << "," << orgW 
            << ") to (" << iH << ", " << iW << ") according to model requirement\n";
        cv::resize(im, im, cv::Size(iW, iH), cv::INTER_CUBIC);
    }

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


void save_predict(std::string savename, int* data, 
        std::vector<unsigned long> insize, 
        std::vector<unsigned long> outsize) {

    std::vector<std::vector<uint8_t>> color_map = get_color_map();
    int oH = outsize[1];
    int oW = outsize[2];
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

    InferenceEngine::Core ie;
    // ie.SetConfig({{CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(YES)}}, "CPU");
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(mdpth);
    InferenceEngine::ExecutableNetwork network = ie.LoadNetwork(model, device);

    auto inp = model.getInputsInfo().begin();
    auto insize = inp->second->getTensorDesc().getDims();
    auto outp = model.getOutputsInfo().begin();
    auto outsize = outp->second->getTensorDesc().getDims();

    std::cout << "----- supported optimizations ----- \n";
    auto cpuOptimizationCapabilities = ie.GetMetric("CPU", METRIC_KEY(OPTIMIZATION_CAPABILITIES)).as<std::vector<std::string>>();
    for (auto &el:cpuOptimizationCapabilities) {
        std::cout << "    " << el << std::endl;
    }

    // std::string enforceBF16 = network.GetConfig(InferenceEngine::PluginConfigParams::KEY_ENFORCE_BF16).as<std::string>();
    // std::cout << "support BF16" << enforceBF16 << std::endl;

    std::cout << "\n----- model info -----\n";
    std::cout << "    model batchsize: " << model.getBatchSize() << std::endl; 
    std::cout << "    input name: " << inp->first << std::endl; 
    std::cout << "    input size: (" 
        << insize[0] << ", " 
        << insize[1] << ", " 
        << insize[2] << ", " 
        << insize[3] << ") \n"; 
    std::cout << "    output name: " << model.getOutputsInfo().begin()->first << std::endl; 
    std::cout << "    output size: (" 
        << outsize[0] << ", " 
        << outsize[1] << ", " 
        << outsize[2] << ") \n"; 
    std::cout << "----------------------\n\n";
}



void test_speed() {

    std::cout << "load network: " << mdpth << std::endl;
    InferenceEngine::Core ie;
    /* if we enforce using bf16 and platform does not support avx512_bf16, then simulation would be used which would drag down speed. If simulation is not supported, there would be exception. */
    // ie.SetConfig({{CONFIG_KEY(ENFORCE_BF16), CONFIG_VALUE(YES)}}, "CPU");
    InferenceEngine::CNNNetwork model = ie.ReadNetwork(mdpth);
    model.setBatchSize(1U);
    InferenceEngine::ExecutableNetwork network = ie.LoadNetwork(model, device);
    InferenceEngine::InferRequest infer_request = network.CreateInferRequest();

    
    std::cout << "test speed ... \n";
    const int n_loops{500};
    auto start = std::chrono::steady_clock::now();
    for (int i{0}; i < n_loops; ++i) {
        infer_request.Infer();
    }
    auto end = std::chrono::steady_clock::now();
    double duration = std::chrono::duration<double, std::milli>(end - start).count();
    duration /= 1000.;
    std::cout << "running " << n_loops << " times, use time: "
        << duration << "s" << std::endl; 
    std::cout << "fps is: " << static_cast<double>(n_loops) / duration << std::endl;
}
