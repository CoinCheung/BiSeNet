

#include "argmax_plugin.h"
#include "kernels.hpp"

#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>


using std::string;
using std::stringstream;
using std::cout;
using std::endl;
using std::vector;


using namespace nvinfer1;

namespace nvinfer1 {

static void CHECK(bool condition, string msg) {
    if (!condition) {
        cout << msg << endl;;
        std::terminate();
    }
}

static void CHECK(bool condition) {
    if (!condition) {
        cout << "assertion fail" << endl;;
        std::terminate();
    }
}


/* 
 * functions of plugin
 *  */
ArgMaxPlugin::ArgMaxPlugin(int64_t axis) : mAxis(axis)
{
}



IPluginCapability* ArgMaxPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept 
{
    try
    {
        if (type == PluginCapabilityType::kBUILD)
        {
            return static_cast<IPluginV3OneBuild*>(this);
        }
        if (type == PluginCapabilityType::kRUNTIME)
        {
            return static_cast<IPluginV3OneRuntime*>(this);
        }
        CHECK(type == PluginCapabilityType::kCORE);
        return static_cast<IPluginV3OneCore*>(this);
    }
    catch (std::exception const& e)
    {
        cout << "[ERROR]: " << e.what() << std::endl;
    }
    return nullptr;
}


IPluginV3* ArgMaxPlugin::clone() noexcept 
{
    try
    {
        auto *plugin = new ArgMaxPlugin(mAxis);
        plugin->setPluginNamespace(mNamespace.c_str());
        return plugin;
    }
    catch (std::exception const& e)
    {
        cout << "[ERROR]: " << e.what() << std::endl;
    }
    return nullptr;
}


char const* ArgMaxPlugin::getPluginName() const noexcept
{
    return "CustomArgMax";
}


char const* ArgMaxPlugin::getPluginVersion() const noexcept 
{
    return "1";
}


char const* ArgMaxPlugin::getPluginNamespace() const noexcept 
{
    return mNamespace.c_str();
}


void ArgMaxPlugin::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}


int32_t ArgMaxPlugin::getNbOutputs() const noexcept 
{
    return 1;
}


int32_t ArgMaxPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
    int32_t nbOutputs) noexcept 
{
    return 0;
}


bool ArgMaxPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept 
{
    stringstream ss;
    ss << "ArgMaxPlugin accepts only two input, but here pos is " << pos;
    CHECK(pos < 2, ss.str());

    // TODO: what if int8
    bool typeOk = inOut[0].desc.type == DataType::kFLOAT || inOut[0].desc.type == DataType::kHALF;

    bool formatOK = inOut[0].desc.format == PluginFormat::kLINEAR;

    return formatOK && typeOk;
}

int32_t ArgMaxPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept
{
    outputTypes[0] = DataType::kINT64;
    return 0;
}


int32_t ArgMaxPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
    int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept 
{

    outputs[0].nbDims = inputs[0].nbDims - 1;

    for (int i{0}; i < outputs[0].nbDims; ++i) {
        if (i < mAxis) {
            outputs[0].d[i] = inputs[0].d[i];
        } else {
            outputs[0].d[i] = inputs[0].d[i + 1];
        }
    }

    return 0;
}


size_t ArgMaxPlugin::getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
    DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept 
{
    return sizeof(mAxis);
}


int32_t ArgMaxPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
    void* const* outputs, void* workspace, cudaStream_t stream) noexcept 
{
    auto type = inputDesc[0].type;

    auto in_dims = inputDesc[0].dims;
    int32_t dimsize = in_dims.d[mAxis];
    int32_t n_size = 1;
    int32_t m_size = 1;
    for (int32_t i{0}; i < in_dims.nbDims; ++i) {
        if (i < mAxis) {
            n_size *= in_dims.d[i];
        } else if (i > mAxis) {
            m_size *= in_dims.d[i];
        }
    }

    string msg("argmax only support fp32 and fp16 currently");
    CHECK ((type == nvinfer1::DataType::kHALF || type == nvinfer1::DataType::kFLOAT), msg);

    if (type == nvinfer1::DataType::kFLOAT) {
        const float* ptr_inp = static_cast<const float*>(inputs[0]);
        int64_t* ptr_out = static_cast<int64_t*>(outputs[0]);
        argMaxFunc<float>(ptr_inp, ptr_out, n_size, dimsize, m_size, &stream);

    } else if (type == nvinfer1::DataType::kHALF) {
        // const __half* ptr_inp = static_cast<const __half*>(inputs[0]);
        // __half* ptr_out = static_cast<__half*>(outputs[0]);
        // argMaxFunc<__half>(ptr_inp, ptr_out, n_size, dimsize, m_size, &stream);

    } else {
    }
    return 0;
}


int32_t ArgMaxPlugin::onShapeChange(
    PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept 
{
    return 0;
}

IPluginV3* ArgMaxPlugin::attachToContext(IPluginResourceContext* context) noexcept 
{
    return clone();
}

PluginFieldCollection const* ArgMaxPlugin::getFieldsToSerialize() noexcept 
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("dim", &mAxis, PluginFieldType::kINT64, 1);
    mFCToSerialize.nbFields = mDataToSerialize.size();
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}


/* 
 * functions of plugin creator
 *  */
ArgMaxPluginCreator::ArgMaxPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back("dim"); // vector<nvinfer1::PluginField>
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}


char const* ArgMaxPluginCreator::getPluginName() const noexcept 
{
    return "CustomArgMax";
}


char const* ArgMaxPluginCreator::getPluginVersion() const noexcept 
{
    return "1";
}


PluginFieldCollection const* ArgMaxPluginCreator::getFieldNames() noexcept 
{
    return &mFC;
}


IPluginV3* ArgMaxPluginCreator::createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept 
{
    try
    {
        int64_t mAxis{0};
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            const string fieldName(fc->fields[i].name);
            if (fieldName == "dim")
            {
                mAxis = *static_cast<int32_t const*>(fc->fields[i].data);
            }
        }
        auto plugin = new ArgMaxPlugin(mAxis);
        plugin->setPluginNamespace(mNamespace.c_str());

        return plugin;
    }
    catch (std::exception const& e)
    {
        cout << "[ERROR]: " << e.what() << std::endl;
    }
    return nullptr;
}


char const* ArgMaxPluginCreator::getPluginNamespace() const noexcept 
{
    return mNamespace.c_str();
}

void ArgMaxPluginCreator::setPluginNamespace(char const* libNamespace) noexcept 
{
    mNamespace = libNamespace;
}


} // namespace nvinfer1 
