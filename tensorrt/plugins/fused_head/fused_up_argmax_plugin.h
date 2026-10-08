#ifndef FUSED_UP_ARGMAX_PLUGIN_H
#define FUSED_UP_ARGMAX_PLUGIN_H

#include <string>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "NvInferRuntimePlugin.h"

namespace fused_head {

constexpr char const* kPLUGIN_NAME = "FusedUpsampleArgMax";
constexpr char const* kPLUGIN_VERSION = "1";
constexpr char const* kPLUGIN_NAMESPACE = "custom";

// tactic = alg * 100000 + rowTile * 10000 + blockSize (block <= 1024, rowTile <= 8, 0 if unused)
enum Alg : int32_t
{
    kALG_SCALAR = 0,       // 1 output pixel / thread, generic scale
    kALG_VEC4 = 1,         // 1x4 output pixels / thread, int4 store, generic scale
    kALG_BLK4X4 = 2,       // 4x4 output pixels / thread, scale==4 fast path
    kALG_BLK4X4_SMEM = 3,  // 4x4 + fp32 shared-memory input tile, scale==4
    kALG_BLKSXR = 4,       // RT x S output tile / thread, scale in {2,4,8}
    kALG_BLKSXR_SMEM = 5,  // ditto + input-dtype shared-memory tile
    kALG_SCALAR_VC = 6,    // ALG0 geometry, 16B channel-vector loads (fp16, 8 adjacent channels)
    kALG_BLKSXR_VC = 7,    // ALG4 geometry, 16B channel-vector loads (fp16, 8 adjacent channels)
    kALG_GROUP_C32 = 8,    // ALG5 geometry, per-32-channel-group smem tile (kCHW32)
    kALG_BLKHW = 9,        // ALG4 for anisotropic / non-power-of-2 scales (SH, SW in {1,2,3,4,6,8,16})
    kALG_BLKHW_VC = 10,    // ditto, 16B channel-vector loads (fp16, 8 adjacent channels)
    kALG_BLKSXR_H2 = 11,   // ALG7 geometry, fp16 inputs interpolated in fp16 (half2), any fp16 layout
    kNB_ALG = 12
};

class FusedUpArgMaxPlugin : public nvinfer1::IPluginV3,
                            public nvinfer1::IPluginV3OneCore,
                            public nvinfer1::IPluginV3OneBuild,
                            public nvinfer1::IPluginV3OneRuntime
{
public:
    FusedUpArgMaxPlugin(FusedUpArgMaxPlugin const&) = default;
    FusedUpArgMaxPlugin(int32_t scaleH, int32_t scaleW);

    // IPluginV3
    nvinfer1::IPluginCapability* getCapabilityInterface(nvinfer1::PluginCapabilityType type) noexcept override;
    nvinfer1::IPluginV3* clone() noexcept override;

    // IPluginV3OneCore
    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* ns) noexcept;

    // IPluginV3OneBuild
    int32_t getNbOutputs() const noexcept override;
    int32_t configurePlugin(nvinfer1::DynamicPluginTensorDesc const* in, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out, int32_t nbOutputs) noexcept override;
    bool supportsFormatCombination(int32_t pos, nvinfer1::DynamicPluginTensorDesc const* inOut, int32_t nbInputs,
        int32_t nbOutputs) noexcept override;
    int32_t getOutputDataTypes(nvinfer1::DataType* outputTypes, int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes, int32_t nbInputs) const noexcept override;
    int32_t getOutputShapes(nvinfer1::DimsExprs const* inputs, int32_t nbInputs,
        nvinfer1::DimsExprs const* shapeInputs, int32_t nbShapeInputs, nvinfer1::DimsExprs* outputs, int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    size_t getWorkspaceSize(nvinfer1::DynamicPluginTensorDesc const* inputs, int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* outputs, int32_t nbOutputs) const noexcept override;

    // --- tactic mechanism ---
    int32_t getNbTactics() noexcept override;
    int32_t getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept override;
    char const* getTimingCacheID() noexcept override;
    char const* getMetadataString() noexcept override;
    int32_t getFormatCombinationLimit() noexcept override;

    // IPluginV3OneRuntime
    int32_t setTactic(int32_t tactic) noexcept override;
    int32_t enqueue(nvinfer1::PluginTensorDesc const* inputDesc, nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override;
    int32_t onShapeChange(nvinfer1::PluginTensorDesc const* in, int32_t nbInputs, nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override;
    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext* context) noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override;

private:
    void buildTacticList();
    void refreshIds();

    int32_t mScaleH{4};
    int32_t mScaleW{4};
    int32_t mC{-1};          // channel count seen in configurePlugin (-1 = unknown)
    int32_t mElemSize{2};    // input element size seen in configurePlugin
    nvinfer1::DataType mInType{nvinfer1::DataType::kHALF};   // input type seen in configurePlugin
    int32_t mTactic{0};
    std::vector<int32_t> mTactics;
    std::vector<nvinfer1::PluginField> mDataToSerialize;
    nvinfer1::PluginFieldCollection mFCToSerialize{};
    std::string mNamespace{kPLUGIN_NAMESPACE};
    std::string mTimingCacheId;
    std::string mMetadata;
};

class FusedUpArgMaxPluginCreator : public nvinfer1::IPluginCreatorV3One
{
public:
    FusedUpArgMaxPluginCreator();
    ~FusedUpArgMaxPluginCreator() override = default;

    char const* getPluginName() const noexcept override;
    char const* getPluginVersion() const noexcept override;
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override;
    nvinfer1::IPluginV3* createPlugin(
        char const* name, nvinfer1::PluginFieldCollection const* fc, nvinfer1::TensorRTPhase phase) noexcept override;
    char const* getPluginNamespace() const noexcept override;
    void setPluginNamespace(char const* ns) noexcept;

private:
    nvinfer1::PluginFieldCollection mFC{};
    std::vector<nvinfer1::PluginField> mAttrs;
    std::string mNamespace{kPLUGIN_NAMESPACE};
};

}  // namespace fused_head

#endif
