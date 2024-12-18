
#ifndef TENSORRT_ARGMAX_PLUGIN_H
#define TENSORRT_ARGMAX_PLUGIN_H

#include <vector>
#include <string>

#include <NvInferRuntimePlugin.h>
#include "NvInferPlugin.h"
#include "NvInfer.h"


using namespace nvinfer1;


namespace nvinfer1 {


class ArgMaxPlugin : public IPluginV3, public IPluginV3OneCore, public IPluginV3OneBuildV2, public IPluginV3OneRuntime
{
public:
    ArgMaxPlugin(ArgMaxPlugin const& p) = default;

    ArgMaxPlugin(int64_t axis);


    // IPluginV3 methods

    IPluginCapability* getCapabilityInterface(PluginCapabilityType type) noexcept override;

    IPluginV3* clone() noexcept override;

    // IPluginV3OneCore methods
    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* pluginNamespace) noexcept;

    // IPluginV3OneBuild methods
    int32_t getNbOutputs() const noexcept override;

    int32_t configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs, DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;

    bool supportsFormatCombination(
        int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;

    int32_t getOutputDataTypes(
        DataType* outputTypes, int32_t nbOutputs, DataType const* inputTypes, int32_t nbInputs) const noexcept override;

    int32_t getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const* shapeInputs,
        int32_t nbShapeInputs, DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& exprBuilder) noexcept override;

    size_t getWorkspaceSize(DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // IPluginV3OneRuntime methods
    int32_t enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc, void const* const* inputs,
        void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;

    int32_t onShapeChange(
        PluginTensorDesc const* in, int32_t nbInputs, PluginTensorDesc const* out, int32_t nbOutputs) noexcept override;

    IPluginV3* attachToContext(IPluginResourceContext* context) noexcept override;

    PluginFieldCollection const* getFieldsToSerialize() noexcept override;


private:
    int64_t mAxis;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize;
    std::string mNamespace;
};




class ArgMaxPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    ArgMaxPluginCreator();

    ~ArgMaxPluginCreator() override = default;

    char const* getPluginName() const noexcept override;

    char const* getPluginVersion() const noexcept override;

    PluginFieldCollection const* getFieldNames() noexcept override;

    IPluginV3* createPlugin(char const* name, PluginFieldCollection const* fc, TensorRTPhase phase) noexcept override;

    char const* getPluginNamespace() const noexcept override;

    void setPluginNamespace(char const* libNamespace) noexcept;


private:
    nvinfer1::PluginFieldCollection mFC;
    std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mNamespace;
};

} // namespace nvinfer1 

#endif

