#include "fused_up_argmax_plugin.h"
#include "fused_up_argmax_kernel.cuh"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <type_traits>
#include <cctype>
#include <cmath>
#include <cstdlib>

using namespace nvinfer1;

namespace fused_head {

namespace {

int32_t const kBLOCKS[] = {128, 256, 512, 1024};
int32_t const kNB_BLOCKS = 4;

inline int32_t algOf(int32_t tactic) { return tactic / 100000; }
inline int32_t rtOf(int32_t tactic) { return (tactic / 10000) % 10; }
inline int32_t blockOf(int32_t tactic) { return tactic % 10000; }
inline int32_t mkTactic(int32_t alg, int32_t rt, int32_t block) { return alg * 100000 + rt * 10000 + block; }

// Row tiles advertised for the RTxS block algorithms, per scale.
inline std::vector<int32_t> rowTilesFor(int32_t S)
{
    if (S == 2) return {1, 2};
    if (S == 4) return {2, 4};
    if (S == 8) return {2, 4, 8};
    return {};
}

// Device properties, only to drop tactics this GPU cannot run; TRT times the rest.
struct DevInfo
{
    int32_t sm{0};
    int32_t l2{0};
    int32_t smemOptin{48 * 1024};
    int32_t major{0};
    int32_t minor{0};
};

DevInfo const& devInfo()
{
    static DevInfo d = [] {
        DevInfo v;
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess)
        {
            cudaGetLastError();   // our own failed query, not the caller's error
            return v;
        }
        cudaDeviceProp pr{};
        bool bad = cudaGetDeviceProperties(&pr, dev) != cudaSuccess;
        if (!bad)
        {
            v.sm = pr.multiProcessorCount;
            v.l2 = pr.l2CacheSize;
            v.major = pr.major;
            v.minor = pr.minor;
        }
        int optin = 48 * 1024;
        if (cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) != cudaSuccess)
        {
            bad = true;
            optin = 48 * 1024;
        }
        v.smemOptin = optin;
        if (bad) cudaGetLastError();
        return v;
    }();
    return d;
}

// Shared-memory footprint of the ALG5 tile, or 0 if the geometry is invalid.
inline size_t smemBytesSxR(int32_t S, int32_t rt, int32_t block, int32_t C, int32_t elemSize)
{
    int32_t nsub = S / rt;
    int32_t tpr = 32 * nsub;
    if (tpr > block) return 0;
    int32_t th = block / tpr;
    return static_cast<size_t>(C) * (th + 2) * 34 * elemSize;
}

// Opt in to >48 KB dynamic shared memory; false -> caller falls back to ALG4.
// Deliberately not memoised (README, "Bug found during bring-up", second bug).
template <typename K>
inline bool ensureSmem(K kernel, size_t bytes)
{
    if (bytes > static_cast<size_t>(devInfo().smemOptin)) return false;
    if (bytes > 48 * 1024)
    {
        if (cudaFuncSetAttribute(reinterpret_cast<void const*>(kernel),
                cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(bytes))
            != cudaSuccess)
        {
            cudaGetLastError();   // swallow the error, caller falls back to ALG4
            return false;
        }
    }
    return true;
}

inline int64_t divUp(int64_t a, int64_t b) { return (a + b - 1) / b; }

template <typename T>
struct NoDeduce
{
    using type = T;
};

// Launch through the runtime API so that a failure of *this* launch is returned
// (and cleared) instead of being mixed up with an earlier, unrelated error.
template <typename... A>
inline bool launchOk(void (*k)(A...), int32_t grid, int32_t block, size_t smem, cudaStream_t s,
    typename NoDeduce<A>::type... args)
{
    void* a[] = {&args...};
    if (cudaLaunchKernel(reinterpret_cast<void const*>(k), dim3(grid), dim3(block), a, smem, s) == cudaSuccess)
        return true;
    cudaGetLastError();
    return false;
}

// Channel-last with 8 adjacent fp16 channels at 16-byte aligned offsets (ALG6/7).
inline bool vec8Layout(BiParams const& p)
{
    return p.nhwc && p.chStride == 1 && p.colStride % 8 == 0 && p.rowStride % 8 == 0 && p.batchStride % 8 == 0;
}

// cap the grid so that grid-stride loops keep the launch cheap for large batch
inline int32_t gridFor(int64_t nThreads, int32_t block)
{
    int64_t g = divUp(nThreads, block);
    return static_cast<int32_t>(std::min<int64_t>(g, 65535 * 16));
}

// ALG4 (RT x S tile per thread) / ALG5 (+ smem tile): bad (S,RT) -1, failed launch -2.
template <typename T, int BLOCK, int S, int RT, int GV>
inline int32_t launchSR(int32_t alg, T const* in, int32_t* out, BiParams const& p, cudaStream_t stream)
{
    if constexpr (!srOk(S, RT))
    {
        return -1;
    }
    else
    {
        constexpr int32_t NSUB = S / RT;
        auto launchGlobal = [&] {
            int64_t n = static_cast<int64_t>(p.N) * p.Hin * NSUB * p.Win;
            return launchOk(kBlockSxR<T, BLOCK, S, RT, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p);
        };
        if (alg == kALG_BLKSXR)
        {
            return launchGlobal() ? 0 : -2;
        }
        if (alg == kALG_GROUP_C32)
        {
            // ALG8: 16-byte channel vectors or 4-byte planes and RT*S <= 16, else ALG4 runs.
            if constexpr (RT * S <= 16 && (GV == 0 || GV == 32 || GV * sizeof(T) == 4))
            {
                constexpr int32_t TH = BLOCK / (32 * NSUB);
                constexpr int KG = (GV == 2 || GV == 4) ? GV : 0;   // kernel's plane mode
                bool vecOk = GV == 32 ? p.colStride == 32
                    : KG > 0          ? true
                                      : (p.colStride * sizeof(T)) % 16 == 0;
                if (p.nhwc && p.chStride == 1 && vecOk)
                {
                    // GV == 32 / 2 / 4: groups / planes of the vector-major layout; otherwise
                    // one pixel holds all channels (channel-last, or vector-major with C <= V)
                    int64_t groupStride = GV > 0 ? p.rowStride * p.Hin : 32;
                    int32_t chStep = GV > 0 ? 0 : 32;
                    size_t smem = groupTileSmemBytes(BLOCK, S, RT, sizeof(T));
                    if (ensureSmem(kGroupTileC32<T, BLOCK, S, RT, KG>, smem))
                    {
                        int32_t tilesX = static_cast<int32_t>(divUp(p.Win, 32));
                        int32_t tilesY = static_cast<int32_t>(divUp(p.Hin, TH));
                        if (launchOk(kGroupTileC32<T, BLOCK, S, RT, KG>, tilesX * tilesY * p.N, BLOCK, smem, stream,
                                in, out, p, tilesX, groupStride, chStep))
                            return 0;
                        return launchGlobal() ? 0 : -2;   // rejected shared-memory launch
                    }
                }
            }
            return launchGlobal() ? 0 : -2;
        }
        if (alg == kALG_BLKSXR_H2)
        {
            // fp16 inputs only, BLOCK 128 / 256 (what is advertised); same register limit as ALG7
            if constexpr (std::is_same<T, __half>::value && RT * S <= 16 && (GV == 0 || GV == 2 || GV == 32)
                && (BLOCK == 128 || BLOCK == 256))
            {
                int64_t n = static_cast<int64_t>(p.N) * p.Hin * NSUB * p.Win;
                return launchOk(kBlockSxRHalf<BLOCK, S, RT, GV>, gridFor(n, BLOCK), BLOCK, 0, stream,
                           reinterpret_cast<__half const*>(in), out, p, vec8Layout(p))
                    ? 0
                    : -2;
            }
            return launchGlobal() ? 0 : -2;
        }
        if (alg == kALG_BLKSXR_VC)
        {
            // ALG6/7: fp16 channel-last within the register budget, else ALG4 runs.
            if constexpr (std::is_same<T, __half>::value && RT * S <= 16 && (GV == 0 || GV == 32))
            {
                if (vec8Layout(p))
                {
                    int64_t n = static_cast<int64_t>(p.N) * p.Hin * NSUB * p.Win;
                    return launchOk(kBlockSxRVecC<BLOCK, S, RT, GV>, gridFor(n, BLOCK), BLOCK, 0, stream,
                               reinterpret_cast<__half const*>(in), out, p)
                        ? 0
                        : -2;
                }
            }
            return launchGlobal() ? 0 : -2;
        }
        if constexpr (BLOCK == 256 || BLOCK == 512)
        {
            constexpr int32_t TW = 32;
            constexpr int32_t TPR = TW * NSUB;
            if constexpr (TPR <= BLOCK)
            {
                constexpr int32_t TH = BLOCK / TPR;
                size_t smem = static_cast<size_t>(p.C) * (TH + 2) * (TW + 2) * sizeof(T);
                if (ensureSmem(kBlockSxRSmem<T, BLOCK, S, RT, GV>, smem))
                {
                    int32_t tilesX = static_cast<int32_t>(divUp(p.Win, TW));
                    int32_t tilesY = static_cast<int32_t>(divUp(p.Hin, TH));
                    // a rejected shared-memory launch is redone with the global-memory variant
                    if (launchOk(kBlockSxRSmem<T, BLOCK, S, RT, GV>, tilesX * tilesY * p.N, BLOCK, smem, stream, in,
                            out, p, tilesX))
                        return 0;
                    return launchGlobal() ? 0 : -2;
                }
            }
        }
        return launchGlobal() ? 0 : -2;   // shared memory does not fit -> global-memory variant
    }
}

template <typename T, int BLOCK, int S, int GV>
inline int32_t launchS(int32_t alg, int32_t rt, T const* in, int32_t* out, BiParams const& p, cudaStream_t stream)
{
    switch (rt)
    {
    case 1: return launchSR<T, BLOCK, S, 1, GV>(alg, in, out, p, stream);
    case 2: return launchSR<T, BLOCK, S, 2, GV>(alg, in, out, p, stream);
    case 4: return launchSR<T, BLOCK, S, 4, GV>(alg, in, out, p, stream);
    case 8: return launchSR<T, BLOCK, S, 8, GV>(alg, in, out, p, stream);
    default: return -1;
    }
}

template <typename T, int BLOCK, int GV>
int32_t launchB(int32_t alg, int32_t rt, int32_t scale, T const* in, int32_t* out, BiParams const& p,
    cudaStream_t stream)
{
    if (alg == kALG_BLKSXR || alg == kALG_BLKSXR_SMEM || alg == kALG_BLKSXR_VC || alg == kALG_GROUP_C32
        || alg == kALG_BLKSXR_H2)
    {
        int32_t rc = -1;
        switch (scale)
        {
        case 2: rc = launchS<T, BLOCK, 2, GV>(alg, rt, in, out, p, stream); break;
        case 4: rc = launchS<T, BLOCK, 4, GV>(alg, rt, in, out, p, stream); break;
        case 8: rc = launchS<T, BLOCK, 8, GV>(alg, rt, in, out, p, stream); break;
        default: break;
        }
        if (rc == 0 || rc == -2) return rc;
        alg = (p.Wout % 4 == 0) ? kALG_VEC4 : kALG_SCALAR;   // unsupported (S,RT)
    }
    switch (alg)
    {
    case kALG_SCALAR:
    {
        int64_t n = static_cast<int64_t>(p.N) * p.Hout * p.Wout;
        if (!launchOk(kScalar<T, BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p)) return -2;
        break;
    }
    case kALG_SCALAR_VC:
    {
        int64_t n = static_cast<int64_t>(p.N) * p.Hout * p.Wout;
        if constexpr (std::is_same<T, __half>::value && (GV == 0 || GV == 32))
        {
            if (vec8Layout(p))
            {
                if (!launchOk(kScalarVecC<BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, reinterpret_cast<__half const*>(in), out, p)) return -2;
                break;
            }
        }
        if (!launchOk(kScalar<T, BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p)) return -2;
        break;
    }
    case kALG_VEC4:
    {
        int64_t n = static_cast<int64_t>(p.N) * p.Hout * (p.Wout / 4);
        if (!launchOk(kVec4<T, BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p)) return -2;
        break;
    }
    case kALG_BLK4X4:
    {
        int64_t n = static_cast<int64_t>(p.N) * p.Hin * p.Win;
        if (!launchOk(kBlock4x4<T, BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p)) return -2;
        break;
    }
    case kALG_BLK4X4_SMEM:
    {
        constexpr int32_t TW = 32;
        constexpr int32_t TH = BLOCK / TW;
        constexpr int32_t IW = TW + 2;
        constexpr int32_t IH = TH + 2;
        size_t smem = static_cast<size_t>(p.C) * IH * IW * sizeof(float);
        if (smem > 48 * 1024)
        {  // shared-memory budget exceeded -> fall back to the global-memory 4x4 kernel
            int64_t n = static_cast<int64_t>(p.N) * p.Hin * p.Win;
            if (!launchOk(kBlock4x4<T, BLOCK, GV>, gridFor(n, BLOCK), BLOCK, 0, stream, in, out, p)) return -2;
            break;
        }
        int32_t tilesX = static_cast<int32_t>(divUp(p.Win, TW));
        int32_t tilesY = static_cast<int32_t>(divUp(p.Hin, TH));
        if (!launchOk(kBlock4x4Smem<T, BLOCK, GV>, tilesX * tilesY * p.N, BLOCK, smem, stream, in, out, p, tilesX)) return -2;
        break;
    }
    default: return -1;
    }
    return 0;
}

// ALG9 / ALG10.  Returns 1 if (SH, SW, RT) was not built (caller falls back).
template <typename T, int GV, int SH, int SW, int RT>
int32_t launchHW3(int32_t alg, T const* in, int32_t* out, BiParams const& p, cudaStream_t stream)
{
    if constexpr (!hwScaleOk(SH, SW) || !hwRtOk(SH, SW, RT))
        return 1;
    else
    {
        int64_t n = static_cast<int64_t>(p.N) * p.Hin * (SH / RT) * p.Win;
        if constexpr (std::is_same<T, __half>::value)
        {
            if constexpr (GV == 0 || GV == 32)
                if (alg == kALG_BLKHW_VC && vec8Layout(p))
                    return launchOk(kBlockHWVecC<128, SH, SW, RT, GV>, gridFor(n, 128), 128, 0, stream, in, out, p) ? 0
                                                                                                            : -2;
        }
        // ALG10 on other layouts runs ALG9 (a duplicate, not a failure)
        return launchOk(kBlockHW<T, 256, SH, SW, RT, GV>, gridFor(n, 256), 256, 0, stream, in, out, p) ? 0 : -2;
    }
}

template <typename T, int GV, int SH, int SW>
int32_t launchHWrt(int32_t alg, int32_t rt, T const* in, int32_t* out, BiParams const& p, cudaStream_t stream)
{
    switch (rt)
    {
    case 1: return launchHW3<T, GV, SH, SW, 1>(alg, in, out, p, stream);
    case 2: return launchHW3<T, GV, SH, SW, 2>(alg, in, out, p, stream);
    case 3: return launchHW3<T, GV, SH, SW, 3>(alg, in, out, p, stream);
    case 4: return launchHW3<T, GV, SH, SW, 4>(alg, in, out, p, stream);
    case 6: return launchHW3<T, GV, SH, SW, 6>(alg, in, out, p, stream);
    case 8: return launchHW3<T, GV, SH, SW, 8>(alg, in, out, p, stream);
    default: return 1;
    }
}

template <typename T, int GV, int SH>
int32_t launchHWsw(int32_t alg, int32_t rt, int32_t sw, T const* in, int32_t* out, BiParams const& p, cudaStream_t stream)
{
    switch (sw)
    {
    case 1: return launchHWrt<T, GV, SH, 1>(alg, rt, in, out, p, stream);
    case 2: return launchHWrt<T, GV, SH, 2>(alg, rt, in, out, p, stream);
    case 3: return launchHWrt<T, GV, SH, 3>(alg, rt, in, out, p, stream);
    case 4: return launchHWrt<T, GV, SH, 4>(alg, rt, in, out, p, stream);
    case 6: return launchHWrt<T, GV, SH, 6>(alg, rt, in, out, p, stream);
    case 8: return launchHWrt<T, GV, SH, 8>(alg, rt, in, out, p, stream);
    case 16: return launchHWrt<T, GV, SH, 16>(alg, rt, in, out, p, stream);
    default: return 1;
    }
}

template <typename T, int GV>
int32_t launchHW(int32_t alg, int32_t rt, int32_t sh, int32_t sw, T const* in, int32_t* out, BiParams const& p,
    cudaStream_t stream)
{
    switch (sh)
    {
    case 1: return launchHWsw<T, GV, 1>(alg, rt, sw, in, out, p, stream);
    case 2: return launchHWsw<T, GV, 2>(alg, rt, sw, in, out, p, stream);
    case 3: return launchHWsw<T, GV, 3>(alg, rt, sw, in, out, p, stream);
    case 4: return launchHWsw<T, GV, 4>(alg, rt, sw, in, out, p, stream);
    case 6: return launchHWsw<T, GV, 6>(alg, rt, sw, in, out, p, stream);
    case 8: return launchHWsw<T, GV, 8>(alg, rt, sw, in, out, p, stream);
    case 16: return launchHWsw<T, GV, 16>(alg, rt, sw, in, out, p, stream);
    default: return 1;
    }
}

template <typename T, int GV>
int32_t launch(int32_t alg, int32_t rt, int32_t block, int32_t scale, T const* in, int32_t* out,
    BiParams const& p, cudaStream_t stream)
{
    switch (block)
    {
    case 128: return launchB<T, 128, GV>(alg, rt, scale, in, out, p, stream);
    case 256: return launchB<T, 256, GV>(alg, rt, scale, in, out, p, stream);
    case 512: return launchB<T, 512, GV>(alg, rt, scale, in, out, p, stream);
    case 1024: return launchB<T, 1024, GV>(alg, rt, scale, in, out, p, stream);
    default: return launchB<T, 256, GV>(alg, rt, scale, in, out, p, stream);
    }
}

// FUSED_HEAD_FORCE_FORMAT (tests, build time): formats to offer, e.g. "chw32"; unset = all.
inline bool hookAllowsFormat(PluginFormat f)
{
    char const* env = getenv("FUSED_HEAD_FORCE_FORMAT");
    if (env == nullptr) return true;
    std::string s;   // case- and blank-insensitive
    for (char const* c = env; *c; ++c)
        if (*c != ' ' && *c != '\t') s += static_cast<char>(std::tolower(static_cast<unsigned char>(*c)));
    if (s.empty()) return true;
    char const* name = "";
    switch (f)
    {
    case PluginFormat::kLINEAR: name = "linear"; break;
    case PluginFormat::kCHW2: name = "chw2"; break;
    case PluginFormat::kHWC8: name = "hwc8"; break;
    case PluginFormat::kCHW4: name = "chw4"; break;
    case PluginFormat::kCHW32: name = "chw32"; break;
    case PluginFormat::kHWC: name = "hwc"; break;
    case PluginFormat::kHWC16: name = "hwc16"; break;
    default: return false;
    }
    std::string list = "," + s + ",";
    return list.find(std::string(",") + name + ",") != std::string::npos;
}

// The (format, type) input pairs TensorRT defines for 4D tensors, minus bf16.
inline bool pairSupported(PluginFormat f, DataType t)
{
    bool f32 = t == DataType::kFLOAT, f16 = t == DataType::kHALF, i8 = t == DataType::kINT8, f8 = t == DataType::kFP8;
    switch (f)
    {
    case PluginFormat::kLINEAR: return f32 || f16 || i8 || f8;
    case PluginFormat::kHWC8: return f16;
    case PluginFormat::kHWC16: return f16 || i8 || f8;
    case PluginFormat::kHWC: return f32;
    case PluginFormat::kCHW2: return f16;
    case PluginFormat::kCHW4: return i8;
    case PluginFormat::kCHW32: return f32 || f16 || i8;
    default: return false;
    }
}

}  // namespace

// =========================================================== plugin
FusedUpArgMaxPlugin::FusedUpArgMaxPlugin(int32_t scaleH, int32_t scaleW)
    : mScaleH(scaleH)
    , mScaleW(scaleW)
{
    buildTacticList();
    refreshIds();
}

void FusedUpArgMaxPlugin::refreshIds()
{
    char buf[256];
    // The advertised tactic list depends on the scale, on C and on the input
    // element size, so all three must be part of the timing-cache key.
    snprintf(buf, sizeof(buf), "FusedUpsampleArgMax_sh%d_sw%d_c%d_e%d%s", mScaleH, mScaleW, mC, mElemSize,
        mInType == DataType::kFP8 ? "_fp8" : "");
    mTimingCacheId = buf;
    snprintf(buf, sizeof(buf), " (half_pixel bilinear + argmax(axis=1) -> int32; sm%d%d, %d SMs, "
                               "L2 %d KB, smem-optin %d KB, %d tactics)",
        devInfo().major, devInfo().minor, devInfo().sm, devInfo().l2 / 1024, devInfo().smemOptin / 1024,
        static_cast<int32_t>(mTactics.size()));
    mMetadata = mTimingCacheId + buf;
}

void FusedUpArgMaxPlugin::buildTacticList()
{
    mTactics.clear();
    // FUSED_HEAD_FORCE_TACTIC=<id>: advertise only that tactic (per-tactic sweeps).
    char const* force = getenv("FUSED_HEAD_FORCE_TACTIC");
    if (force != nullptr && force[0] != '\0')
    {
        int32_t t = atoi(force);
        if (t > 0)
        {
            mTactics.push_back(t);
            return;
        }
    }
    bool square = (mScaleH == mScaleW);
    int32_t S = mScaleH;
    bool scale4 = (square && S == 4);
    bool blkS = square && (S == 2 || S == 4 || S == 8);

    for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_SCALAR, 0, kBLOCKS[b]));
    for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_VEC4, 0, kBLOCKS[b]));
    // The list must not depend on the format, so layout-specific variants are always
    // advertised; on other layouts they run the generic kernel of the same geometry.
    bool vecC = (mElemSize == 2);   // fp16 only; the layout is not known here
    if (vecC)
        for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_SCALAR_VC, 0, kBLOCKS[b]));
    if (scale4)
    {
        for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_BLK4X4, 0, kBLOCKS[b]));
        for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_BLK4X4_SMEM, 0, kBLOCKS[b]));
    }
    if (blkS)
    {
        for (int32_t rt : rowTilesFor(S))
        {
            for (int32_t b = 0; b < kNB_BLOCKS; ++b) mTactics.push_back(mkTactic(kALG_BLKSXR, rt, kBLOCKS[b]));
        }
        if (vecC)
        {
            for (int32_t rt : rowTilesFor(S))
            {
                if (rt * S > 16) continue;   // would spill; the launcher falls back to ALG4 anyway
                for (int32_t b = 0; b < kNB_BLOCKS; ++b)
                    mTactics.push_back(mkTactic(kALG_BLKSXR_VC, rt, kBLOCKS[b]));
            }
        }
        // ALG5 only exists for block sizes 256/512 (the tile geometry needs
        // 32 * S/RT <= block and at least two tile rows to be worth staging).
        for (int32_t rt : rowTilesFor(S))
        {
            for (int32_t b : {256, 512})
            {
                size_t need = smemBytesSxR(S, rt, b, mC > 0 ? mC : 19, mElemSize);
                // the only device-dependent filter: a tile that cannot fit would run ALG4
                if (need == 0 || need > static_cast<size_t>(devInfo().smemOptin)) continue;
                mTactics.push_back(mkTactic(kALG_BLKSXR_SMEM, rt, b));
            }
        }
        // ALG8: 16-byte channel vectors or 4-byte planes, else ALG4; ALG7's register limit.
        for (int32_t rt : rowTilesFor(S))
        {
            if (rt * S > 16) continue;
            for (int32_t b : {128, 256, 512})
            {
                size_t need = groupTileSmemBytes(b, S, rt, mElemSize);
                if (need > static_cast<size_t>(devInfo().smemOptin)) continue;
                mTactics.push_back(mkTactic(kALG_GROUP_C32, rt, b));
            }
        }
    }
    // ALG11: fp16 inputs, interpolation in fp16 (square 2 / 4 / 8, RT*S <= 16 like ALG7).
    // Test hook FUSED_HEAD_FP32_ONLY=1 leaves it out (tests that assert fp32 labels).
    char const* fp32Only = getenv("FUSED_HEAD_FP32_ONLY");
    if (blkS && vecC && !(fp32Only != nullptr && fp32Only[0] == '1'))
        for (int32_t rt : rowTilesFor(S))
        {
            if (rt * S > 16) continue;
            for (int32_t b : {128, 256}) mTactics.push_back(mkTactic(kALG_BLKSXR_H2, rt, b));
        }
    // ALG9 / ALG10: the scale pairs the square kernels above do not cover
    if (hwScaleOk(mScaleH, mScaleW))
        for (int32_t rt : {1, 2, 3, 4, 6, 8})
        {
            if (!hwRtOk(mScaleH, mScaleW, rt)) continue;
            mTactics.push_back(mkTactic(kALG_BLKHW, rt, 256));
            if (vecC) mTactics.push_back(mkTactic(kALG_BLKHW_VC, rt, 128));
        }
    if (mTactics.empty()) mTactics.push_back(mkTactic(kALG_SCALAR, 0, 256));
}

IPluginCapability* FusedUpArgMaxPlugin::getCapabilityInterface(PluginCapabilityType type) noexcept
{
    if (type == PluginCapabilityType::kBUILD) return static_cast<IPluginV3OneBuild*>(this);
    if (type == PluginCapabilityType::kRUNTIME) return static_cast<IPluginV3OneRuntime*>(this);
    return static_cast<IPluginV3OneCore*>(this);
}

IPluginV3* FusedUpArgMaxPlugin::clone() noexcept
{
    auto* p = new (std::nothrow) FusedUpArgMaxPlugin(*this);
    if (p) p->setPluginNamespace(mNamespace.c_str());
    return p;
}

char const* FusedUpArgMaxPlugin::getPluginName() const noexcept { return kPLUGIN_NAME; }
char const* FusedUpArgMaxPlugin::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
char const* FusedUpArgMaxPlugin::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedUpArgMaxPlugin::setPluginNamespace(char const* ns) noexcept { mNamespace = ns ? ns : ""; }
int32_t FusedUpArgMaxPlugin::getNbOutputs() const noexcept { return 1; }

int32_t FusedUpArgMaxPlugin::configurePlugin(DynamicPluginTensorDesc const* in, int32_t nbInputs,
    DynamicPluginTensorDesc const*, int32_t) noexcept
{
    // C and element size decide which smem tactics fit; configurePlugin precedes tactic queries.
    if (in != nullptr && nbInputs > 0 && in[0].desc.dims.nbDims == 4)
    {
        mC = static_cast<int32_t>(in[0].desc.dims.d[1]);
        mInType = in[0].desc.type;
        switch (in[0].desc.type)
        {
        case DataType::kFLOAT: mElemSize = 4; break;
        case DataType::kHALF: mElemSize = 2; break;
        case DataType::kINT8: mElemSize = 1; break;
        case DataType::kFP8: mElemSize = 1; break;
        default: mElemSize = 2; break;
        }
        buildTacticList();
        refreshIds();
    }
    return 0;
}

bool FusedUpArgMaxPlugin::supportsFormatCombination(
    int32_t pos, DynamicPluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    if (pos == 1)
    {
        return inOut[1].desc.type == DataType::kINT32 && inOut[1].desc.format == PluginFormat::kLINEAR;
    }
    auto t = inOut[0].desc.type;
    auto f = inOut[0].desc.format;
    if (!hookAllowsFormat(f))
    {
        return false;
    }
    // Every pair is offered; TensorRT keeps the fastest (usually what the producer
    // writes natively).  Vector-major layouts (kCHW2/4/32) require a static C.
    bool vecMajor = f == PluginFormat::kCHW2 || f == PluginFormat::kCHW4 || f == PluginFormat::kCHW32;
    bool staticC = inOut[0].desc.dims.nbDims == 4 && inOut[0].desc.dims.d[1] > 0;
    return pairSupported(f, t) && (staticC || !vecMajor);
}

int32_t FusedUpArgMaxPlugin::getOutputDataTypes(
    DataType* outputTypes, int32_t, DataType const*, int32_t) const noexcept
{
    outputTypes[0] = DataType::kINT32;
    return 0;
}

int32_t FusedUpArgMaxPlugin::getOutputShapes(DimsExprs const* inputs, int32_t nbInputs, DimsExprs const*, int32_t,
    DimsExprs* outputs, int32_t nbOutputs, IExprBuilder& eb) noexcept
{
    if (inputs[0].nbDims != 4) return -1;
    outputs[0].nbDims = 3;
    outputs[0].d[0] = inputs[0].d[0];
    outputs[0].d[1] = eb.operation(DimensionOperation::kPROD, *inputs[0].d[2], *eb.constant(mScaleH));
    outputs[0].d[2] = eb.operation(DimensionOperation::kPROD, *inputs[0].d[3], *eb.constant(mScaleW));
    return 0;
}

size_t FusedUpArgMaxPlugin::getWorkspaceSize(
    DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) const noexcept
{
    return 0;
}

int32_t FusedUpArgMaxPlugin::getNbTactics() noexcept { return static_cast<int32_t>(mTactics.size()); }

int32_t FusedUpArgMaxPlugin::getValidTactics(int32_t* tactics, int32_t nbTactics) noexcept
{
    if (nbTactics != static_cast<int32_t>(mTactics.size())) return -1;
    std::copy(mTactics.begin(), mTactics.end(), tactics);
    return 0;
}

char const* FusedUpArgMaxPlugin::getTimingCacheID() noexcept { return mTimingCacheId.c_str(); }
char const* FusedUpArgMaxPlugin::getMetadataString() noexcept { return mMetadata.c_str(); }
int32_t FusedUpArgMaxPlugin::getFormatCombinationLimit() noexcept { return 32; }

int32_t FusedUpArgMaxPlugin::setTactic(int32_t tactic) noexcept
{
    mTactic = tactic;
    return 0;
}

int32_t FusedUpArgMaxPlugin::onShapeChange(PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) noexcept
{
    return 0;
}

IPluginV3* FusedUpArgMaxPlugin::attachToContext(IPluginResourceContext*) noexcept { return clone(); }

PluginFieldCollection const* FusedUpArgMaxPlugin::getFieldsToSerialize() noexcept
{
    mDataToSerialize.clear();
    mDataToSerialize.emplace_back("scale_h", &mScaleH, PluginFieldType::kINT32, 1);
    mDataToSerialize.emplace_back("scale_w", &mScaleW, PluginFieldType::kINT32, 1);
    mFCToSerialize.nbFields = static_cast<int32_t>(mDataToSerialize.size());
    mFCToSerialize.fields = mDataToSerialize.data();
    return &mFCToSerialize;
}

int32_t FusedUpArgMaxPlugin::enqueue(PluginTensorDesc const* inputDesc, PluginTensorDesc const* outputDesc,
    void const* const* inputs, void* const* outputs, void*, cudaStream_t stream) noexcept
{
    auto const& d = inputDesc[0].dims;
    if (d.nbDims != 4 || d.d[1] <= 0 || !pairSupported(inputDesc[0].format, inputDesc[0].type)) return -1;
    if (d.d[0] == 0 || d.d[2] == 0 || d.d[3] == 0) return 0;   // empty tensor: nothing to write

    BiParams p{};
    p.N = static_cast<int32_t>(d.d[0]);
    p.C = static_cast<int32_t>(d.d[1]);
    p.Hin = static_cast<int32_t>(d.d[2]);
    p.Win = static_cast<int32_t>(d.d[3]);
    p.Hout = p.Hin * mScaleH;
    p.Wout = p.Win * mScaleW;
    p.invSh = 1.f / static_cast<float>(mScaleH);
    p.invSw = 1.f / static_cast<float>(mScaleW);
    auto const type = inputDesc[0].type;
    auto const fmt = inputDesc[0].format;
    p.dq = 1.f;   // argmax does not depend on the per-tensor scale; int8 at 2^k scales stays exact

    // channel-last: C padded to cpad; vector-major: groups of vec channels,
    // [N][C/vec][H][W][vec], one group == channel-last padded to vec
    int64_t cpad = 0, vec = 0;
    switch (fmt)
    {
    case PluginFormat::kHWC8: cpad = 8; break;
    case PluginFormat::kHWC16: cpad = 16; break;
    case PluginFormat::kHWC: cpad = 1; break;
    case PluginFormat::kCHW2: vec = 2; break;
    case PluginFormat::kCHW4: vec = 4; break;
    case PluginFormat::kCHW32: vec = 32; break;
    default: break;
    }
    if (cpad > 0)
    {
        cpad = ((p.C + cpad - 1) / cpad) * cpad;
        p.nhwc = 1;
        p.chStride = 1;
        p.colStride = cpad;
        p.rowStride = static_cast<int64_t>(p.Win) * cpad;
        p.batchStride = static_cast<int64_t>(p.Hin) * p.Win * cpad;
    }
    else if (vec > 0)
    {
        int64_t groups = (p.C + vec - 1) / vec;
        p.nhwc = 1;
        p.chStride = 1;
        p.colStride = vec;
        p.rowStride = static_cast<int64_t>(p.Win) * vec;
        p.batchStride = groups * p.Hin * p.Win * vec;
    }
    else
    {
        p.nhwc = 0;
        p.chStride = static_cast<int64_t>(p.Hin) * p.Win;
        p.colStride = 1;
        p.rowStride = p.Win;
        p.batchStride = static_cast<int64_t>(p.C) * p.Hin * p.Win;
    }

    bool square = (mScaleH == mScaleW);
    bool scale4 = (square && mScaleH == 4);
    bool blkS = square && (mScaleH == 2 || mScaleH == 4 || mScaleH == 8);

    // Tactic 0 is TensorRT's reserved "default tactic"; anything we do not
    // recognise also falls back to the default rather than failing the launch.
    int32_t tactic = mTactic;
    int32_t alg = blkS ? kALG_BLKSXR : kALG_VEC4;
    int32_t rt = (mScaleH == 8) ? 2 : (mScaleH == 4 ? 4 : 2);
    int32_t block = 256;
    if (tactic != 0)
    {
        int32_t a = algOf(tactic);
        int32_t r = rtOf(tactic);
        int32_t b = blockOf(tactic);
        bool okAlg = (a >= 0 && a < kNB_ALG);
        bool okBlk = (b == 128 || b == 256 || b == 512 || b == 1024);
        if (okAlg && okBlk)
        {
            alg = a;
            rt = r;
            block = b;
        }
    }

    // Guards: fall back to a universally-valid algorithm if preconditions fail.
    if ((alg == kALG_BLKSXR || alg == kALG_BLKSXR_SMEM || alg == kALG_BLKSXR_VC || alg == kALG_GROUP_C32
            || alg == kALG_BLKSXR_H2)
        && !blkS)
        alg = kALG_VEC4;
    if ((alg == kALG_BLK4X4 || alg == kALG_BLK4X4_SMEM) && !scale4) alg = kALG_VEC4;
    if ((alg == kALG_BLKHW || alg == kALG_BLKHW_VC) && !(hwScaleOk(mScaleH, mScaleW) && hwRtOk(mScaleH, mScaleW, rt)))
        alg = kALG_VEC4;
    if (alg == kALG_VEC4 && (p.Wout % 4 != 0)) alg = kALG_SCALAR;

    int32_t* out = static_cast<int32_t*>(outputs[0]);
    // a vector-major layout with more than one channel group needs grouped addressing
    int64_t gv = (vec > 0 && p.C > vec) ? vec : 0;
    auto run = [&](auto tag, auto gvTag) {
        using T = decltype(tag);
        constexpr int G = decltype(gvTag)::value;
        T const* in = static_cast<T const*>(inputs[0]);
        if (alg == kALG_BLKHW || alg == kALG_BLKHW_VC)
        {
            int32_t r = launchHW<T, G>(alg, rt, mScaleH, mScaleW, in, out, p, stream);
            if (r != 1) return r;   // 1: (SH, SW, RT) not built; the guard above prevents it
            return launch<T, G>((p.Wout % 4 == 0) ? kALG_VEC4 : kALG_SCALAR, 0, block, mScaleH, in, out, p, stream);
        }
        return launch<T, G>(alg, rt, block, mScaleH, in, out, p, stream);
    };
    using G0 = std::integral_constant<int, 0>;
    using G2 = std::integral_constant<int, 2>;
    using G4 = std::integral_constant<int, 4>;
    using G32 = std::integral_constant<int, 32>;
    int32_t rc = -1;
    switch (type)
    {
    case DataType::kFLOAT: rc = gv == 32 ? run(float{}, G32{}) : run(float{}, G0{}); break;
    case DataType::kHALF:
        rc = gv == 32 ? run(__half{}, G32{}) : gv == 2 ? run(__half{}, G2{}) : run(__half{}, G0{});
        break;
    case DataType::kINT8:
        rc = gv == 32 ? run(int8_t{}, G32{}) : gv == 4 ? run(int8_t{}, G4{}) : run(int8_t{}, G0{});
        break;
    case DataType::kFP8: rc = run(__nv_fp8_e4m3{}, G0{}); break;
    default: break;
    }
    return rc == 0 ? 0 : -1;
}

// =========================================================== creator
FusedUpArgMaxPluginCreator::FusedUpArgMaxPluginCreator()
{
    mAttrs.clear();
    mAttrs.emplace_back("scale_h", nullptr, PluginFieldType::kINT32, 1);
    mAttrs.emplace_back("scale_w", nullptr, PluginFieldType::kINT32, 1);
    mFC.nbFields = static_cast<int32_t>(mAttrs.size());
    mFC.fields = mAttrs.data();
}

char const* FusedUpArgMaxPluginCreator::getPluginName() const noexcept { return kPLUGIN_NAME; }
char const* FusedUpArgMaxPluginCreator::getPluginVersion() const noexcept { return kPLUGIN_VERSION; }
PluginFieldCollection const* FusedUpArgMaxPluginCreator::getFieldNames() noexcept { return &mFC; }
char const* FusedUpArgMaxPluginCreator::getPluginNamespace() const noexcept { return mNamespace.c_str(); }
void FusedUpArgMaxPluginCreator::setPluginNamespace(char const* ns) noexcept { mNamespace = ns ? ns : ""; }

IPluginV3* FusedUpArgMaxPluginCreator::createPlugin(
    char const*, PluginFieldCollection const* fc, TensorRTPhase) noexcept
{
    int32_t sh = 4, sw = 4;
    if (fc)
    {
        for (int32_t i = 0; i < fc->nbFields; ++i)
        {
            std::string name(fc->fields[i].name);
            void const* d = fc->fields[i].data;
            if (d == nullptr) continue;
            if (name != "scale_h" && name != "scale_w") continue;
            // The ONNX parser may hand us INT32 / INT64 / FLOAT depending on how the
            // attribute was written, so accept all numeric encodings.
            double v = 0;   // every accepted encoding is exact in a double
            switch (fc->fields[i].type)
            {
            case PluginFieldType::kINT32: v = *static_cast<int32_t const*>(d); break;
            case PluginFieldType::kINT64: v = static_cast<double>(*static_cast<int64_t const*>(d)); break;
            case PluginFieldType::kINT16: v = *static_cast<int16_t const*>(d); break;
            case PluginFieldType::kINT8: v = *static_cast<int8_t const*>(d); break;
            case PluginFieldType::kFLOAT32: v = *static_cast<float const*>(d); break;
            case PluginFieldType::kFLOAT64: v = *static_cast<double const*>(d); break;
            default: continue;
            }
            // 2.5, inf, NaN and values outside int32 are not valid scales
            if (!(v >= 1.0 && v <= 2147483647.0) || v != std::floor(v)) return nullptr;
            if (name == "scale_h") sh = static_cast<int32_t>(v); else sw = static_cast<int32_t>(v);
        }
    }
    if (sh <= 0 || sw <= 0) return nullptr;   // invalid attribute: fail the import visibly
    auto* p = new (std::nothrow) FusedUpArgMaxPlugin(sh, sw);
    if (p) p->setPluginNamespace(mNamespace.c_str());
    return p;
}

// =========================================================== registration
namespace {
class Registrar
{
public:
    Registrar()
    {
        auto* reg = getPluginRegistry();
        if (reg)
        {
            static FusedUpArgMaxPluginCreator creator;
            reg->registerCreator(creator, kPLUGIN_NAMESPACE);
        }
    }
};
Registrar gRegistrar;
}  // namespace

}  // namespace fused_head

// Plugin-library entry points (setPluginsToSerialize / loadLibrary);
// TRT_PLUGIN_STATIC_ONLY omits them when several plugins share one binary.
#ifndef TRT_PLUGIN_STATIC_ONLY
extern "C" void setLoggerFinder(nvinfer1::ILoggerFinder*) {}

extern "C" nvinfer1::IPluginCreatorInterface* const* getCreators(int32_t& nbCreators)
{
    static fused_head::FusedUpArgMaxPluginCreator sCreator;
    static nvinfer1::IPluginCreatorInterface* sCreators[] = {&sCreator};
    nbCreators = 1;
    return sCreators;
}
#endif
