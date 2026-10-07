// Fused bilinear-upsample + ArgMax CUDA kernels: ONNX Resize (linear, half_pixel)
// -> ArgMax (axis 1, first index on ties) -> int32, all C channels interpolated in
// fp32 before the argmax.  Semantics and algorithms: README "Semantics", "Tactics".
#pragma once

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace fused_head {

// ---------------------------------------------------------------- load helpers
// raw read-only load for fp8 (CUDA has no __ldg overload for it)
using ::__ldg;
__device__ __forceinline__ __nv_fp8_e4m3 __ldg(__nv_fp8_e4m3 const* p)
{
    __nv_fp8_e4m3 v;
    v.__x = ::__ldg(reinterpret_cast<unsigned char const*>(p));
    return v;
}

template <typename T>
struct Cvt;

template <>
struct Cvt<float> {
    static __device__ __forceinline__ float load(float const* p, float dq) { return __ldg(p); }
    static __device__ __forceinline__ float cvt(float v, float dq) { return v; }
};
template <>
struct Cvt<__half> {
    static __device__ __forceinline__ float load(__half const* p, float dq) { return __half2float(__ldg(p)); }
    static __device__ __forceinline__ float cvt(__half v, float dq) { return __half2float(v); }
};
template <>
struct Cvt<int8_t> {
    static __device__ __forceinline__ float load(int8_t const* p, float dq)
    {
        return static_cast<float>(__ldg(p)) * dq;
    }
    static __device__ __forceinline__ float cvt(int8_t v, float dq) { return static_cast<float>(v) * dq; }
};
template <>
struct Cvt<__nv_fp8_e4m3> {
    static __device__ __forceinline__ float cvt(__nv_fp8_e4m3 v, float dq) { return static_cast<float>(v) * dq; }
    static __device__ __forceinline__ float load(__nv_fp8_e4m3 const* p, float dq) { return cvt(__ldg(p), dq); }
};

// Explicit strides let one kernel body serve NCHW, channel-last and CHW2/4/32
// (README "Addressing").
struct BiParams
{
    int32_t N, C, Hin, Win, Hout, Wout;
    int64_t batchStride, chStride, rowStride, colStride;
    float invSh, invSw;
    float dq;        // input scale; the plugin passes 1.f (argmax ignores it)
    int32_t nhwc;    // 1 if input is channel-last
};

#define FH_NEG_INF (-3.402823466e+38f)

// half_pixel source coordinate, clamped to [0, inSize-1]
__device__ __forceinline__ void srcCoord(int dst, float invS, int inSize, int& i0, int& i1, float& w)
{
    float s = (static_cast<float>(dst) + 0.5f) * invS - 0.5f;
    s = fminf(fmaxf(s, 0.f), static_cast<float>(inSize - 1));
    i0 = static_cast<int>(s);   // s >= 0, so trunc == floor
    if (i0 > inSize - 1) i0 = inSize - 1;
    i1 = min(i0 + 1, inSize - 1);
    w = s - static_cast<float>(i0);
}

// Offset of channel c: GV > 0 vector-major group + lane, GV == 0 c * chStride.
template <int GV>
__device__ __forceinline__ int64_t fhChOff(int c, BiParams const& p)
{
    if constexpr (GV > 0)
        return static_cast<int64_t>(static_cast<unsigned>(c) / GV) * p.rowStride * p.Hin + static_cast<unsigned>(c) % GV;
    else
        return c * p.chStride;
}

// Constant weights of the scale==4 / half_pixel fast path (see kBlock4x4).
__constant__ float kW4[4] = {0.625f, 0.875f, 0.125f, 0.375f};

// ============================================================================
// ALG 0 : scalar -- one output pixel per thread, generic scales.
// ============================================================================
template <typename T, int BLOCK, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kScalar(T const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int64_t total = static_cast<int64_t>(p.N) * p.Hout * p.Wout;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int x = static_cast<int>(i % p.Wout);
        int64_t t = i / p.Wout;
        int y = static_cast<int>(t % p.Hout);
        int n = static_cast<int>(t / p.Hout);

        int y0, y1, x0, x1;
        float wy, wx;
        srcCoord(y, p.invSh, p.Hin, y0, y1, wy);
        srcCoord(x, p.invSw, p.Win, x0, x1, wx);

        T const* base = in + n * p.batchStride;
        int64_t o00 = y0 * p.rowStride + x0 * p.colStride;
        int64_t o01 = y0 * p.rowStride + x1 * p.colStride;
        int64_t o10 = y1 * p.rowStride + x0 * p.colStride;
        int64_t o11 = y1 * p.rowStride + x1 * p.colStride;

        float best = FH_NEG_INF;
        int bi = 0;
        for (int c = 0; c < p.C; ++c)
        {
            T const* pl = base + fhChOff<GV>(c, p);
            float a = Cvt<T>::load(pl + o00, p.dq);
            float b = Cvt<T>::load(pl + o01, p.dq);
            float cc = Cvt<T>::load(pl + o10, p.dq);
            float d = Cvt<T>::load(pl + o11, p.dq);
            float v = (1.f - wy) * ((1.f - wx) * a + wx * b) + wy * ((1.f - wx) * cc + wx * d);
            if (v > best) { best = v; bi = c; }
        }
        out[i] = bi;
    }
}

// ============================================================================
// ALG 1 : vec4 -- 4 consecutive output x per thread, single 16B int4 store.
//         Generic scales.  Requires Wout % 4 == 0.
// ============================================================================
template <typename T, int BLOCK, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kVec4(T const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int qw = p.Wout >> 2;
    int64_t total = static_cast<int64_t>(p.N) * p.Hout * qw;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int q = static_cast<int>(i % qw);
        int64_t t = i / qw;
        int y = static_cast<int>(t % p.Hout);
        int n = static_cast<int>(t / p.Hout);

        int y0, y1;
        float wy;
        srcCoord(y, p.invSh, p.Hin, y0, y1, wy);

        int64_t r0 = y0 * p.rowStride;
        int64_t r1 = y1 * p.rowStride;
        int64_t ox0[4], ox1[4];
        float wx[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            int a0, a1;
            srcCoord(4 * q + j, p.invSw, p.Win, a0, a1, wx[j]);
            ox0[j] = a0 * p.colStride;
            ox1[j] = a1 * p.colStride;
        }

        T const* base = in + n * p.batchStride;
        float best[4];
        int bi[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) { best[j] = FH_NEG_INF; bi[j] = 0; }

        for (int c = 0; c < p.C; ++c)
        {
            T const* pl = base + fhChOff<GV>(c, p);
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                float a = Cvt<T>::load(pl + r0 + ox0[j], p.dq);
                float b = Cvt<T>::load(pl + r0 + ox1[j], p.dq);
                float cc = Cvt<T>::load(pl + r1 + ox0[j], p.dq);
                float d = Cvt<T>::load(pl + r1 + ox1[j], p.dq);
                float v = (1.f - wy) * ((1.f - wx[j]) * a + wx[j] * b)
                        + wy * ((1.f - wx[j]) * cc + wx[j] * d);
                if (v > best[j]) { best[j] = v; bi[j] = c; }
            }
        }
        reinterpret_cast<int4*>(out)[i] = make_int4(bi[0], bi[1], bi[2], bi[3]);
    }
}

// ============================================================================
// ============================================================================
// ALG 2 : 4x4 output block per thread, scale 4: a 3x3 neighbourhood with constant
//         weights, 9 loads per channel for 16 outputs (README "Algorithm").
// ============================================================================
#define FH_BILERP_16(V)                                                              \
    _Pragma("unroll") for (int r = 0; r < 4; ++r)                                    \
    {                                                                                \
        int ro = (r < 2) ? 0 : 1;                                                    \
        float wy = kW4[r];                                                           \
        float omw = 1.f - wy;                                                        \
        float t0 = omw * V[ro][0] + wy * V[ro + 1][0];                               \
        float t1 = omw * V[ro][1] + wy * V[ro + 1][1];                               \
        float t2 = omw * V[ro][2] + wy * V[ro + 1][2];                               \
        _Pragma("unroll") for (int s = 0; s < 4; ++s)                                \
        {                                                                            \
            float wx = kW4[s];                                                       \
            float a = (s < 2) ? t0 : t1;                                             \
            float b = (s < 2) ? t1 : t2;                                             \
            float val = (1.f - wx) * a + wx * b;                                     \
            int kk = r * 4 + s;                                                      \
            if (val > best[kk]) { best[kk] = val; bi[kk] = c; }                      \
        }                                                                            \
    }

#define FH_STORE_4X4()                                                                       \
    {                                                                                        \
        int64_t obase = (static_cast<int64_t>(n) * p.Hout + 4 * pp) * p.Wout + 4 * q;         \
        _Pragma("unroll") for (int r = 0; r < 4; ++r)                                        \
        {                                                                                    \
            *reinterpret_cast<int4*>(out + obase + static_cast<int64_t>(r) * p.Wout)          \
                = make_int4(bi[r * 4 + 0], bi[r * 4 + 1], bi[r * 4 + 2], bi[r * 4 + 3]);      \
        }                                                                                    \
    }

template <typename T, int BLOCK, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlock4x4(T const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int qw = p.Win;   // Wout / 4
    int qh = p.Hin;   // Hout / 4
    int64_t total = static_cast<int64_t>(p.N) * qh * qw;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;

    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int q = static_cast<int>(i % qw);
        int64_t t = i / qw;
        int pp = static_cast<int>(t % qh);
        int n = static_cast<int>(t / qh);

        int64_t ry[3] = {max(pp - 1, 0) * p.rowStride, pp * p.rowStride, min(pp + 1, p.Hin - 1) * p.rowStride};
        int64_t cx[3] = {max(q - 1, 0) * p.colStride, q * p.colStride, min(q + 1, p.Win - 1) * p.colStride};

        T const* base = in + n * p.batchStride;
        float best[16];
        int bi[16];
#pragma unroll
        for (int k = 0; k < 16; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

        for (int c = 0; c < p.C; ++c)
        {
            T const* pl = base + fhChOff<GV>(c, p);
            float v[3][3];
#pragma unroll
            for (int k = 0; k < 3; ++k)
            {
                v[0][k] = Cvt<T>::load(pl + ry[0] + cx[k], p.dq);
                v[1][k] = Cvt<T>::load(pl + ry[1] + cx[k], p.dq);
                v[2][k] = Cvt<T>::load(pl + ry[2] + cx[k], p.dq);
            }
            FH_BILERP_16(v)
        }
        FH_STORE_4X4()
    }
}

// ============================================================================
// ============================================================================
// ALG 3 : ALG 2 with the input tile staged in shared memory, layout [iy][ix][c].
// ============================================================================
template <typename T, int BLOCK, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlock4x4Smem(
    T const* __restrict__ in, int32_t* __restrict__ out, BiParams p, int tilesX)
{
    extern __shared__ float shTile[];
    constexpr int TW = 32;
    constexpr int TH = BLOCK / TW;
    constexpr int IW = TW + 2;
    constexpr int IH = TH + 2;

    int tid = threadIdx.x;
    int rest = blockIdx.x / tilesX;
    int tileX = blockIdx.x - rest * tilesX;
    int nTilesY = (p.Hin + TH - 1) / TH;
    int n = rest / nTilesY;
    int tileY = rest - n * nTilesY;

    int qx0 = tileX * TW;
    int qy0 = tileY * TH;

    T const* base = in + n * p.batchStride;
    int C = p.C;
    int total = IH * IW * C;

    if (p.nhwc)
    {
        // channel-fastest global reads -> coalesced
        for (int e = tid; e < total; e += BLOCK)
        {
            int px = e / C;
            int c = e - px * C;
            int iy = px / IW;
            int ix = px - iy * IW;
            int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
            int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
            shTile[e] = Cvt<T>::load(base + gy * p.rowStride + gx * p.colStride + fhChOff<GV>(c, p), p.dq);
        }
    }
    else
    {
        // x-fastest global reads -> coalesced for NCHW
        int perCh = IH * IW;
        for (int e = tid; e < total; e += BLOCK)
        {
            int c = e / perCh;
            int px = e - c * perCh;
            int iy = px / IW;
            int ix = px - iy * IW;
            int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
            int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
            shTile[px * C + c] = Cvt<T>::load(base + gy * p.rowStride + gx * p.colStride + fhChOff<GV>(c, p), p.dq);
        }
    }
    __syncthreads();

    int lx = tid % TW;
    int ly = tid / TW;
    int q = qx0 + lx;
    int pp = qy0 + ly;
    if (q >= p.Win || pp >= p.Hin) return;

    float best[16];
    int bi[16];
#pragma unroll
    for (int k = 0; k < 16; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

    float const* s0 = shTile + (static_cast<int>(ly + 0) * IW + lx) * C;
    float const* s1 = shTile + (static_cast<int>(ly + 1) * IW + lx) * C;
    float const* s2 = shTile + (static_cast<int>(ly + 2) * IW + lx) * C;

    for (int c = 0; c < C; ++c)
    {
        float v[3][3];
#pragma unroll
        for (int k = 0; k < 3; ++k)
        {
            v[0][k] = s0[k * C + c];
            v[1][k] = s1[k * C + c];
            v[2][k] = s2[k * C + c];
        }
        FH_BILERP_16(v)
    }
    FH_STORE_4X4()
}

// ============================================================================
// ============================================================================
// ALG 4 / 5 : constant-weight blocks for any square integer scale S; a thread owns
// RT x S outputs from a 3x3 neighbourhood, RT is a tactic (README "alg 4 / alg 5").
// ============================================================================

// Fractional interpolation weight of sub-pixel r under half_pixel with scale S.
// Called only with literal r inside fully unrolled loops, so it folds away.
template <int S>
__device__ __host__ __forceinline__ constexpr float sWeight(int r)
{
    return ((static_cast<float>(r) + 0.5f) / static_cast<float>(S) - 0.5f)
        + ((r < S / 2) ? 1.0f : 0.0f);
}

constexpr bool srOk(int S, int RT)
{
    return (S == 2 && (RT == 1 || RT == 2)) || (S == 4 && (RT == 2 || RT == 4))
        || (S == 8 && (RT == 2 || RT == 4 || RT == 8));
}

// horizontal lerp + argmax update for one output row; v0/v1/v2 are the three
// already vertically-interpolated column samples.
#define FH_ROW_ARGMAX(S, r, t0, t1, t2)                                                    \
    _Pragma("unroll") for (int s = 0; s < S; ++s)                                          \
    {                                                                                      \
        float wx = sWeight<S>(s);                                                          \
        float aa = (s < S / 2) ? t0 : t1;                                                  \
        float bb = (s < S / 2) ? t1 : t2;                                                  \
        float val = (1.f - wx) * aa + wx * bb;                                             \
        int kk = (r) * S + s;                                                              \
        if (val > best[kk]) { best[kk] = val; bi[kk] = c; }                                \
    }

template <int S, int RT>
__device__ __forceinline__ void fhStoreTile(int32_t* __restrict__ out, int64_t obase, int32_t Wout,
    int const (&bi)[RT * S])
{
#pragma unroll
    for (int r = 0; r < RT; ++r)
    {
        int32_t* o = out + obase + static_cast<int64_t>(r) * Wout;
        if constexpr (S % 4 == 0)
        {
#pragma unroll
            for (int j = 0; j < S / 4; ++j)
                *reinterpret_cast<int4*>(o + 4 * j)
                    = make_int4(bi[r * S + 4 * j], bi[r * S + 4 * j + 1], bi[r * S + 4 * j + 2],
                        bi[r * S + 4 * j + 3]);
        }
        else if constexpr (S % 2 == 0)
        {
#pragma unroll
            for (int j = 0; j < S / 2; ++j)
                *reinterpret_cast<int2*>(o + 2 * j) = make_int2(bi[r * S + 2 * j], bi[r * S + 2 * j + 1]);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < S; ++j) o[j] = bi[r * S + j];
        }
    }
}

// ---------------------------------------------------------------- ALG 4
// One thread owns an RT x S output tile; input read straight from global (L1).
template <typename T, int BLOCK, int S, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockSxR(T const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    constexpr int NSUB = S / RT;              // sub-tiles per S-row output group
    constexpr int NR = (RT == S) ? 3 : 2;     // input rows actually needed
    constexpr int NOUT = RT * S;

    int qw = p.Win;
    int64_t total = static_cast<int64_t>(p.N) * p.Hin * NSUB * qw;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;

    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int q = static_cast<int>(i % qw);
        int64_t t = i / qw;
        int sub = static_cast<int>(t % NSUB);
        t /= NSUB;
        int pp = static_cast<int>(t % p.Hin);
        int n = static_cast<int>(t / p.Hin);
        int rBase = sub * RT;

        int64_t ry[NR];
        if constexpr (NR == 3)
        {
            ry[0] = static_cast<int64_t>(max(pp - 1, 0)) * p.rowStride;
            ry[1] = static_cast<int64_t>(pp) * p.rowStride;
            ry[2] = static_cast<int64_t>(min(pp + 1, p.Hin - 1)) * p.rowStride;
        }
        else
        {
            int b0 = (rBase < S / 2) ? pp - 1 : pp;
            ry[0] = static_cast<int64_t>(min(max(b0, 0), p.Hin - 1)) * p.rowStride;
            ry[1] = static_cast<int64_t>(min(max(b0 + 1, 0), p.Hin - 1)) * p.rowStride;
        }
        int64_t cx[3] = {static_cast<int64_t>(max(q - 1, 0)) * p.colStride,
            static_cast<int64_t>(q) * p.colStride,
            static_cast<int64_t>(min(q + 1, p.Win - 1)) * p.colStride};

        // vertical weights: rBase is a runtime value when NSUB > 1, so these are
        // hoisted out of the channel loop instead of being folded as literals.
        float wyv[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
        {
            float o = (static_cast<float>(rBase + r) + 0.5f) * (1.0f / static_cast<float>(S)) - 0.5f;
            wyv[r] = (o < 0.f) ? o + 1.f : o;
        }

        T const* base = in + n * p.batchStride;
        float best[NOUT];
        int bi[NOUT];
#pragma unroll
        for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

        for (int c = 0; c < p.C; ++c)
        {
            T const* pl = base + fhChOff<GV>(c, p);
            float v[NR][3];
#pragma unroll
            for (int a = 0; a < NR; ++a)
            {
#pragma unroll
                for (int k = 0; k < 3; ++k) v[a][k] = Cvt<T>::load(pl + ry[a] + cx[k], p.dq);
            }
#pragma unroll
            for (int r = 0; r < RT; ++r)
            {
                constexpr int roC = 0;
                int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : roC;
                float wy = wyv[r];
                float omw = 1.f - wy;
                float t0 = omw * v[ro][0] + wy * v[ro + 1][0];
                float t1 = omw * v[ro][1] + wy * v[ro + 1][1];
                float t2 = omw * v[ro][2] + wy * v[ro + 1][2];
                FH_ROW_ARGMAX(S, r, t0, t1, t2)
            }
        }
        int64_t obase = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(S) * pp + rBase)
                * p.Wout
            + static_cast<int64_t>(S) * q;
        fhStoreTile<S, RT>(out, obase, p.Wout, bi);
    }
}

// ---------------------------------------------------------------- ALG 5
// ---------------------------------------------------------------- ALG 5
// ALG 4 with the input tile in shared memory, in the input dtype (bit-exact).
// Tile: 32 input columns x BLOCK / (32 * S/RT) rows.
template <typename T, int BLOCK, int S, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockSxRSmem(
    T const* __restrict__ in, int32_t* __restrict__ out, BiParams p, int tilesX)
{
    extern __shared__ __align__(16) unsigned char shRaw[];
    T* shTile = reinterpret_cast<T*>(shRaw);

    constexpr int TW = 32;
    constexpr int NSUB = S / RT;
    constexpr int TPR = TW * NSUB;            // threads per input tile row
    constexpr int TH = BLOCK / TPR;
    constexpr int IW = TW + 2;
    constexpr int IH = TH + 2;
    constexpr int NR = (RT == S) ? 3 : 2;
    constexpr int NOUT = RT * S;

    int tid = threadIdx.x;
    int rest = blockIdx.x / tilesX;
    int tileX = blockIdx.x - rest * tilesX;
    int nTilesY = (p.Hin + TH - 1) / TH;
    int n = rest / nTilesY;
    int tileY = rest - n * nTilesY;

    int qx0 = tileX * TW;
    int qy0 = tileY * TH;

    T const* base = in + n * p.batchStride;
    int C = p.C;
    int total = IH * IW * C;

    if (p.nhwc)
    {
        for (int e = tid; e < total; e += BLOCK)
        {
            int px = e / C;
            int c = e - px * C;
            int iy = px / IW;
            int ix = px - iy * IW;
            int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
            int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
            shTile[e] = __ldg(base + gy * p.rowStride + gx * p.colStride + fhChOff<GV>(c, p));
        }
    }
    else
    {
        int perCh = IH * IW;
        for (int e = tid; e < total; e += BLOCK)
        {
            int c = e / perCh;
            int px = e - c * perCh;
            int iy = px / IW;
            int ix = px - iy * IW;
            int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
            int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
            shTile[px * C + c] = __ldg(base + gy * p.rowStride + gx * p.colStride + fhChOff<GV>(c, p));
        }
    }
    __syncthreads();

    int lx = tid % TW;
    int r2 = tid / TW;
    int sub = r2 % NSUB;
    int ly = r2 / NSUB;
    int q = qx0 + lx;
    int pp = qy0 + ly;
    if (q >= p.Win || pp >= p.Hin) return;
    int rBase = sub * RT;

    // shared-memory row of the first input row this thread needs
    int iy0 = (NR == 3) ? ly : ((rBase < S / 2) ? ly : ly + 1);

    float wyv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r)
    {
        float o = (static_cast<float>(rBase + r) + 0.5f) * (1.0f / static_cast<float>(S)) - 0.5f;
        wyv[r] = (o < 0.f) ? o + 1.f : o;
    }

    float best[NOUT];
    int bi[NOUT];
#pragma unroll
    for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

    T const* sr[NR];
#pragma unroll
    for (int a = 0; a < NR; ++a) sr[a] = shTile + (static_cast<int>(iy0 + a) * IW + lx) * C;

    for (int c = 0; c < C; ++c)
    {
        float v[NR][3];
#pragma unroll
        for (int a = 0; a < NR; ++a)
        {
#pragma unroll
            for (int k = 0; k < 3; ++k) v[a][k] = Cvt<T>::cvt(sr[a][k * C + c], p.dq);
        }
#pragma unroll
        for (int r = 0; r < RT; ++r)
        {
            int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : 0;
            float wy = wyv[r];
            float omw = 1.f - wy;
            float t0 = omw * v[ro][0] + wy * v[ro + 1][0];
            float t1 = omw * v[ro][1] + wy * v[ro + 1][1];
            float t2 = omw * v[ro][2] + wy * v[ro + 1][2];
            FH_ROW_ARGMAX(S, r, t0, t1, t2)
        }
    }
    int64_t obase
        = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(S) * pp + rBase) * p.Wout
        + static_cast<int64_t>(S) * q;
    fhStoreTile<S, RT>(out, obase, p.Wout, bi);
}


// ============================================================================
// ============================================================================
// ALG 6 / 7 : 16-byte channel loads, fp16 with 8 adjacent channels (kHWC8/16, kCHW32);
// other layouts fall back to the scalar algorithm.  Same arithmetic, bit-identical to
// ALG 0 / 4 (README addendum, section 2).
// ============================================================================

// 8 fp16 channels (16 B) -> 8 fp32; 16-byte aligned in kHWC8 / kCHW32.
__device__ __forceinline__ void fhLoadH8(__half const* p, float* dst)
{
    float4 raw = __ldg(reinterpret_cast<float4 const*>(p));
    __half2 const* h = reinterpret_cast<__half2 const*>(&raw);
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        float2 f = __half22float2(h[j]);
        dst[2 * j] = f.x;
        dst[2 * j + 1] = f.y;
    }
}

// ---------------------------------------------------------------- ALG 6
// ALG 0's geometry (one output pixel per thread) with 16-byte channel loads.
template <int BLOCK, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kScalarVecC(
    __half const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int64_t total = static_cast<int64_t>(p.N) * p.Hout * p.Wout;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int x = static_cast<int>(i % p.Wout);
        int64_t t = i / p.Wout;
        int y = static_cast<int>(t % p.Hout);
        int n = static_cast<int>(t / p.Hout);

        int y0, y1, x0, x1;
        float wy, wx;
        srcCoord(y, p.invSh, p.Hin, y0, y1, wy);
        srcCoord(x, p.invSw, p.Win, x0, x1, wx);

        __half const* base = in + n * p.batchStride;
        int64_t o00 = y0 * p.rowStride + x0 * p.colStride;
        int64_t o01 = y0 * p.rowStride + x1 * p.colStride;
        int64_t o10 = y1 * p.rowStride + x0 * p.colStride;
        int64_t o11 = y1 * p.rowStride + x1 * p.colStride;

        float best = FH_NEG_INF;
        int bi = 0;
        int c = 0;
        float a[8], b[8], cv[8], d[8];
        for (; c + 8 <= p.C; c += 8)
        {
            __half const* pl;   // 8 lanes never straddle a group
            if constexpr (GV > 0) pl = base + fhChOff<GV>(c, p); else pl = base + c;
            fhLoadH8(pl + o00, a);
            fhLoadH8(pl + o01, b);
            fhLoadH8(pl + o10, cv);
            fhLoadH8(pl + o11, d);
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                float v = (1.f - wy) * ((1.f - wx) * a[j] + wx * b[j])
                    + wy * ((1.f - wx) * cv[j] + wx * d[j]);
                if (v > best) { best = v; bi = c + j; }
            }
        }
        for (; c < p.C; ++c)   // C % 8 tail, scalar so no padding lane is read
        {
            __half const* pl;
            if constexpr (GV > 0) pl = base + fhChOff<GV>(c, p); else pl = base + c;
            float av = __half2float(__ldg(pl + o00));
            float bv = __half2float(__ldg(pl + o01));
            float cc = __half2float(__ldg(pl + o10));
            float dv = __half2float(__ldg(pl + o11));
            float v = (1.f - wy) * ((1.f - wx) * av + wx * bv) + wy * ((1.f - wx) * cc + wx * dv);
            if (v > best) { best = v; bi = c; }
        }
        out[i] = bi;
    }
}

// ---------------------------------------------------------------- ALG 7
// ---------------------------------------------------------------- ALG 7
// ALG 4's geometry with 16-byte channel loads; RT*S <= 16, beyond that it spills.
template <int BLOCK, int S, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockSxRVecC(
    __half const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    constexpr int NSUB = S / RT;
    constexpr int NR = (RT == S) ? 3 : 2;
    constexpr int NOUT = RT * S;

    int qw = p.Win;
    int64_t total = static_cast<int64_t>(p.N) * p.Hin * NSUB * qw;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;

    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int q = static_cast<int>(i % qw);
        int64_t t = i / qw;
        int sub = static_cast<int>(t % NSUB);
        t /= NSUB;
        int pp = static_cast<int>(t % p.Hin);
        int n = static_cast<int>(t / p.Hin);
        int rBase = sub * RT;

        int64_t ry[NR];
        if constexpr (NR == 3)
        {
            ry[0] = static_cast<int64_t>(max(pp - 1, 0)) * p.rowStride;
            ry[1] = static_cast<int64_t>(pp) * p.rowStride;
            ry[2] = static_cast<int64_t>(min(pp + 1, p.Hin - 1)) * p.rowStride;
        }
        else
        {
            int b0 = (rBase < S / 2) ? pp - 1 : pp;
            ry[0] = static_cast<int64_t>(min(max(b0, 0), p.Hin - 1)) * p.rowStride;
            ry[1] = static_cast<int64_t>(min(max(b0 + 1, 0), p.Hin - 1)) * p.rowStride;
        }
        int64_t cx[3] = {static_cast<int64_t>(max(q - 1, 0)) * p.colStride,
            static_cast<int64_t>(q) * p.colStride,
            static_cast<int64_t>(min(q + 1, p.Win - 1)) * p.colStride};

        float wyv[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
        {
            float o = (static_cast<float>(rBase + r) + 0.5f) * (1.0f / static_cast<float>(S)) - 0.5f;
            wyv[r] = (o < 0.f) ? o + 1.f : o;
        }

        __half const* base = in + n * p.batchStride;
        float best[NOUT];
        int bi[NOUT];
#pragma unroll
        for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

        int cb = 0;
        float vv[NR][3][8];
        for (; cb + 8 <= p.C; cb += 8)
        {
            __half const* pl;
            if constexpr (GV > 0) pl = base + fhChOff<GV>(cb, p); else pl = base + cb;
#pragma unroll
            for (int aa2 = 0; aa2 < NR; ++aa2)
            {
#pragma unroll
                for (int k = 0; k < 3; ++k) fhLoadH8(pl + ry[aa2] + cx[k], vv[aa2][k]);
            }
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                int c = cb + j;      // FH_ROW_ARGMAX writes bi[kk] = c
#pragma unroll
                for (int r = 0; r < RT; ++r)
                {
                    int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : 0;
                    float wy = wyv[r];
                    float omw = 1.f - wy;
                    float t0 = omw * vv[ro][0][j] + wy * vv[ro + 1][0][j];
                    float t1 = omw * vv[ro][1][j] + wy * vv[ro + 1][1][j];
                    float t2 = omw * vv[ro][2][j] + wy * vv[ro + 1][2][j];
                    FH_ROW_ARGMAX(S, r, t0, t1, t2)
                }
            }
        }
        for (int c = cb; c < p.C; ++c)
        {
            __half const* pl;
            if constexpr (GV > 0) pl = base + fhChOff<GV>(c, p); else pl = base + c;
            float v[NR][3];
#pragma unroll
            for (int aa2 = 0; aa2 < NR; ++aa2)
            {
#pragma unroll
                for (int k = 0; k < 3; ++k) v[aa2][k] = __half2float(__ldg(pl + ry[aa2] + cx[k]));
            }
#pragma unroll
            for (int r = 0; r < RT; ++r)
            {
                int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : 0;
                float wy = wyv[r];
                float omw = 1.f - wy;
                float t0 = omw * v[ro][0] + wy * v[ro + 1][0];
                float t1 = omw * v[ro][1] + wy * v[ro + 1][1];
                float t2 = omw * v[ro][2] + wy * v[ro + 1][2];
                FH_ROW_ARGMAX(S, r, t0, t1, t2)
            }
        }
        int64_t obase = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(S) * pp + rBase)
                * p.Wout
            + static_cast<int64_t>(S) * q;
        fhStoreTile<S, RT>(out, obase, p.Wout, bi);
    }
}

// ---------------------------------------------------------------- ALG 8
// ---------------------------------------------------------------- ALG 8
// One 32-channel group at a time staged in shared memory: 16-byte loads (kCHW32,
// channel-last with 16-byte pixels) or 4-byte planes (GV 2 / 4: kCHW2 fp16, kCHW4
// int8); ALG 5's arithmetic (README "ALG8").
template <typename T, int BLOCK, int S, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kGroupTileC32(
    T const* __restrict__ in, int32_t* __restrict__ out, BiParams p, int tilesX, int64_t groupStride, int chStep)
{
    constexpr int TW = 32;
    constexpr int NSUB = S / RT;
    constexpr int TH = BLOCK / (TW * NSUB);
    constexpr int IW = TW + 2;
    constexpr int IH = TH + 2;
    constexpr int NR = (RT == S) ? 3 : 2;
    constexpr int NOUT = RT * S;
    constexpr int VEC = 16 / sizeof(T);       // channels per 16-byte vector
    constexpr int NV = 32 / VEC;              // vectors per pixel of one group
    constexpr int PITCH = 32 + VEC;           // pixel pitch in elements

    extern __shared__ __align__(16) unsigned char shRaw[];
    T* sh = reinterpret_cast<T*>(shRaw);

    int tid = threadIdx.x;
    int rest = blockIdx.x / tilesX;
    int tileX = blockIdx.x - rest * tilesX;
    int nTilesY = (p.Hin + TH - 1) / TH;
    int n = rest / nTilesY;
    int tileY = rest - n * nTilesY;
    int qx0 = tileX * TW;
    int qy0 = tileY * TH;

    int lx = tid % TW;
    int r2 = tid / TW;
    int sub = r2 % NSUB;
    int ly = r2 / NSUB;
    int q = qx0 + lx;
    int pp = qy0 + ly;
    bool active = q < p.Win && pp < p.Hin;
    int rBase = sub * RT;
    int iy0 = (NR == 3) ? ly : ((rBase < S / 2) ? ly : ly + 1);

    float wyv[RT];
#pragma unroll
    for (int r = 0; r < RT; ++r)
    {
        float o = (static_cast<float>(rBase + r) + 0.5f) * (1.0f / static_cast<float>(S)) - 0.5f;
        wyv[r] = (o < 0.f) ? o + 1.f : o;
    }
    float best[NOUT];
    int bi[NOUT];
#pragma unroll
    for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

    T const* base = in + n * p.batchStride;
    int groups = (p.C + 31) / 32;

    for (int g = 0; g < groups; ++g)
    {
        if constexpr (GV == 2 || GV == 4)
        {
            static_assert(GV * sizeof(T) == 4, "4-byte planes only");
            constexpr int NP = 32 / GV;                                    // planes per group
            int np = min(NP, (p.C + GV - 1) / GV - g * NP);                // planes that exist
            T const* gb = base + static_cast<int64_t>(g) * NP * groupStride;
            // one thread: one pixel, 4 planes (4-byte loads, each coalesced across the warp),
            // one 16-byte shared store (pixel pitch 5 / 3 x 16 bytes: no bank conflict)
            for (int e = tid; e < IH * IW * (NP / 4); e += BLOCK)
            {
                int kq = e / (IH * IW);
                int px = e - kq * (IH * IW);
                int k0 = kq * 4;
                if (k0 >= np) continue;
                int iy = px / IW;
                int ix = px - iy * IW;
                int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
                int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
                T const* src = gb + gy * p.rowStride + gx * GV;
                uint32_t w[4];
#pragma unroll
                for (int m = 0; m < 4; ++m)   // planes past the last one are never evaluated
                    w[m] = k0 + m < np ? __ldg(reinterpret_cast<uint32_t const*>(src + (k0 + m) * groupStride)) : 0u;
                *reinterpret_cast<uint4*>(sh + px * PITCH + k0 * GV) = make_uint4(w[0], w[1], w[2], w[3]);
            }
        }
        else
        {
        T const* gb = base + g * groupStride;
        int nv = min(NV, static_cast<int>(p.colStride - g * chStep) / VEC);   // vectors inside the pixel
        for (int e = tid; e < IH * IW * NV; e += BLOCK)
        {
            int px = e / NV;
            int v = e - px * NV;
            if (v >= nv) continue;
            int iy = px / IW;
            int ix = px - iy * IW;
            int gy = min(max(qy0 + iy - 1, 0), p.Hin - 1);
            int gx = min(max(qx0 + ix - 1, 0), p.Win - 1);
            *reinterpret_cast<uint4*>(sh + px * PITCH + v * VEC)
                = __ldg(reinterpret_cast<uint4 const*>(gb + gy * p.rowStride + gx * p.colStride + v * VEC));
        }
        }
        __syncthreads();

        if (active)
        {
            int cn = min(32, p.C - g * 32);
#pragma unroll 1
            for (int vb = 0; vb < NV; ++vb)
            {
                if (vb * VEC >= cn) break;
                uint4 raw[NR][3];
#pragma unroll
                for (int a = 0; a < NR; ++a)
                {
#pragma unroll
                    for (int k = 0; k < 3; ++k)
                        raw[a][k] = *reinterpret_cast<uint4 const*>(
                            sh + ((iy0 + a) * IW + lx + k) * PITCH + vb * VEC);
                }
#pragma unroll
                for (int j = 0; j < VEC; ++j)
                {
                    if (vb * VEC + j >= cn) break;
                    int c = g * 32 + vb * VEC + j;   // FH_ROW_ARGMAX writes bi[kk] = c
                    float vv[NR][3];
#pragma unroll
                    for (int a = 0; a < NR; ++a)
                    {
#pragma unroll
                        for (int k = 0; k < 3; ++k)
                            vv[a][k] = Cvt<T>::cvt(reinterpret_cast<T const*>(&raw[a][k])[j], p.dq);
                    }
#pragma unroll
                    for (int r = 0; r < RT; ++r)
                    {
                        int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : 0;
                        float wy = wyv[r];
                        float omw = 1.f - wy;
                        float t0 = omw * vv[ro][0] + wy * vv[ro + 1][0];
                        float t1 = omw * vv[ro][1] + wy * vv[ro + 1][1];
                        float t2 = omw * vv[ro][2] + wy * vv[ro + 1][2];
                        FH_ROW_ARGMAX(S, r, t0, t1, t2)
                    }
                }
            }
        }
        __syncthreads();
    }
    if (!active) return;
    int64_t obase = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(S) * pp + rBase) * p.Wout
        + static_cast<int64_t>(S) * q;
    fhStoreTile<S, RT>(out, obase, p.Wout, bi);
}

// ALG8 dynamic shared memory: (TH+2) x 34 pixels x (32 channels + 16 bytes)
inline size_t groupTileSmemBytes(int block, int S, int rt, int elemSize)
{
    return static_cast<size_t>(block / (32 * (S / rt)) + 2) * 34 * (32 + 16 / elemSize) * elemSize;
}


// ---------------------------------------------------------------- ALG 11
// ---------------------------------------------------------------- ALG 11
// ALG 7's geometry with the interpolation in fp16 (half2); labels may differ from
// fp32 where two channels are within fp16 rounding (README "ALG11").
template <int BLOCK, int S, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockSxRHalf(__half const* __restrict__ in, int32_t* __restrict__ out, BiParams p, bool vec8)
{
    constexpr int NSUB = S / RT;
    constexpr int NR = (RT == S) ? 3 : 2;
    constexpr int NOUT = RT * S;
    int64_t total = static_cast<int64_t>(p.N) * p.Hin * NSUB * p.Win;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        int q = static_cast<int>(i % p.Win);
        int64_t t = i / p.Win;
        int sub = static_cast<int>(t % NSUB);
        t /= NSUB;
        int pp = static_cast<int>(t % p.Hin);
        int n = static_cast<int>(t / p.Hin);
        int rBase = sub * RT;
        int64_t ry[NR];
        if constexpr (NR == 3)
        {
            ry[0] = static_cast<int64_t>(max(pp - 1, 0)) * p.rowStride;
            ry[1] = static_cast<int64_t>(pp) * p.rowStride;
            ry[2] = static_cast<int64_t>(min(pp + 1, p.Hin - 1)) * p.rowStride;
        }
        else
        {
            int b0 = (rBase < S / 2) ? pp - 1 : pp;
            ry[0] = static_cast<int64_t>(min(max(b0, 0), p.Hin - 1)) * p.rowStride;
            ry[1] = static_cast<int64_t>(min(max(b0 + 1, 0), p.Hin - 1)) * p.rowStride;
        }
        int64_t cx[3] = {static_cast<int64_t>(max(q - 1, 0)) * p.colStride, static_cast<int64_t>(q) * p.colStride,
            static_cast<int64_t>(min(q + 1, p.Win - 1)) * p.colStride};
        __half2 wy2[RT], omw2[RT];
#pragma unroll
        for (int r = 0; r < RT; ++r)
        {
            float o = (static_cast<float>(rBase + r) + 0.5f) * (1.0f / static_cast<float>(S)) - 0.5f;
            float w = (o < 0.f) ? o + 1.f : o;   // k / 2S: exact in fp16
            wy2[r] = __float2half2_rn(w);
            omw2[r] = __float2half2_rn(1.f - w);
        }
        __half const* base = in + n * p.batchStride;
        float best[NOUT];
        int bi[NOUT];
#pragma unroll
        for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }

        // one channel pair: v = NR x 3 taps, lane x = channel c, lane y = channel c + 1
        auto pair = [&](int c, __half2 const (&v)[NR][3], bool two) {
#pragma unroll
            for (int r = 0; r < RT; ++r)
            {
                int ro = (NR == 3) ? ((r < S / 2) ? 0 : 1) : 0;
                __half2 tt[3];
#pragma unroll
                for (int k = 0; k < 3; ++k) tt[k] = __hfma2(wy2[r], v[ro + 1][k], __hmul2(omw2[r], v[ro][k]));
#pragma unroll
                for (int s = 0; s < S; ++s)
                {
                    __half2 wx2 = __float2half2_rn(sWeight<S>(s));
                    __half2 om2 = __float2half2_rn(1.f - sWeight<S>(s));
                    __half2 aa = (s < S / 2) ? tt[0] : tt[1];
                    __half2 bb = (s < S / 2) ? tt[1] : tt[2];
                    float2 f = __half22float2(__hfma2(wx2, bb, __hmul2(om2, aa)));
                    int kk = r * S + s;
                    if (f.x > best[kk]) { best[kk] = f.x; bi[kk] = c; }
                    if (two && f.y > best[kk]) { best[kk] = f.y; bi[kk] = c + 1; }
                }
            }
        };

        int cb = 0;
        if constexpr (GV == 0 || GV == 32)
        {
            if (vec8)
                for (; cb + 8 <= p.C; cb += 8)
                {
                    __half const* pl = base + fhChOff<GV>(cb, p);
                    uint4 raw[NR][3];
#pragma unroll
                    for (int a = 0; a < NR; ++a)
#pragma unroll
                        for (int k = 0; k < 3; ++k) raw[a][k] = __ldg(reinterpret_cast<uint4 const*>(pl + ry[a] + cx[k]));
#pragma unroll
                    for (int m = 0; m < 4; ++m)
                    {
                        __half2 v[NR][3];
#pragma unroll
                        for (int a = 0; a < NR; ++a)
#pragma unroll
                            for (int k = 0; k < 3; ++k) v[a][k] = reinterpret_cast<__half2 const*>(&raw[a][k])[m];
                        pair(cb + 2 * m, v, true);
                    }
                }
        }
        for (; cb < p.C; cb += 2)
        {
            bool two = cb + 1 < p.C;
            __half const* p0 = base + fhChOff<GV>(cb, p);
            __half const* p1 = base + fhChOff<GV>(two ? cb + 1 : cb, p);
            __half2 v[NR][3];
#pragma unroll
            for (int a = 0; a < NR; ++a)
#pragma unroll
                for (int k = 0; k < 3; ++k) v[a][k] = __halves2half2(__ldg(p0 + ry[a] + cx[k]), __ldg(p1 + ry[a] + cx[k]));
            pair(cb, v, two);
        }
        int64_t obase = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(S) * pp + rBase) * p.Wout
            + static_cast<int64_t>(S) * q;
        fhStoreTile<S, RT>(out, obase, p.Wout, bi);
    }
}

// ---------------------------------------------------------------- ALG 9 / 10
// ---------------------------------------------------------------- ALG 9 / 10
// ALG 4 / 7 for scales the square {2,4,8} kernels miss (anisotropic, non-power-of-2,
// 16), from each pixel's 3x3 neighbourhood (README "ALG9 / ALG10").
template <int S>
__device__ __host__ __forceinline__ constexpr float hwWeight(int r)
{
    return (2 * r + 1 < S) ? static_cast<float>(static_cast<double>(2 * r + 1 + S) / (2.0 * S))
                           : static_cast<float>(static_cast<double>(2 * r + 1 - S) / (2.0 * S));
}

// vertical weight / side of the runtime sub-row R (static search, folds to selects)
template <int SH>
__device__ __forceinline__ void hwRow(int R, float& w, bool& up)
{
    constexpr float kw[16] = {hwWeight<SH>(0), hwWeight<SH>(1), hwWeight<SH>(2), hwWeight<SH>(3),
        hwWeight<SH>(4), hwWeight<SH>(5), hwWeight<SH>(6), hwWeight<SH>(7), hwWeight<SH>(8), hwWeight<SH>(9),
        hwWeight<SH>(10), hwWeight<SH>(11), hwWeight<SH>(12), hwWeight<SH>(13), hwWeight<SH>(14), hwWeight<SH>(15)};
    w = 0.f;
    up = false;
#pragma unroll
    for (int k = 0; k < SH; ++k)
        if (k == R) { w = kw[k]; up = 2 * k + 1 < SH; }
}

template <int SW, int RT>
__device__ __forceinline__ void fhStoreTileW(int32_t* __restrict__ out, int64_t obase, int32_t Wout,
    int const (&bi)[RT * SW])
{
#pragma unroll
    for (int r = 0; r < RT; ++r)
    {
        int32_t* o = out + obase + static_cast<int64_t>(r) * Wout;
        if constexpr (SW % 4 == 0)
        {
#pragma unroll
            for (int j = 0; j < SW / 4; ++j)
                *reinterpret_cast<int4*>(o + 4 * j) = make_int4(bi[r * SW + 4 * j], bi[r * SW + 4 * j + 1],
                    bi[r * SW + 4 * j + 2], bi[r * SW + 4 * j + 3]);
        }
        else if constexpr (SW % 2 == 0)
        {
#pragma unroll
            for (int j = 0; j < SW / 2; ++j)
                *reinterpret_cast<int2*>(o + 2 * j) = make_int2(bi[r * SW + 2 * j], bi[r * SW + 2 * j + 1]);
        }
        else
        {
#pragma unroll
            for (int j = 0; j < SW; ++j) o[j] = bi[r * SW + j];
        }
    }
}

// one channel: v = 3 x 3 neighbourhood (rows pp-1..pp+1, columns q-1..q+1, clamped)
#define FH_HW_CHANNEL(c, v)                                                                 \
    _Pragma("unroll") for (int r = 0; r < RT; ++r)                                          \
    {                                                                                       \
        float wy = wyv[r];                                                                  \
        float omw = 1.f - wy;                                                               \
        float t[3];                                                                         \
        _Pragma("unroll") for (int k = 0; k < 3; ++k)                                       \
        {                                                                                   \
            float a = upv[r] ? v[0][k] : v[1][k];                                           \
            float b = upv[r] ? v[1][k] : v[2][k];                                           \
            t[k] = omw * a + wy * b;                                                        \
        }                                                                                   \
        _Pragma("unroll") for (int s = 0; s < SW; ++s)                                      \
        {                                                                                   \
            constexpr float kw[16] = {hwWeight<SW>(0), hwWeight<SW>(1), hwWeight<SW>(2),    \
                hwWeight<SW>(3), hwWeight<SW>(4), hwWeight<SW>(5), hwWeight<SW>(6),         \
                hwWeight<SW>(7), hwWeight<SW>(8), hwWeight<SW>(9), hwWeight<SW>(10),        \
                hwWeight<SW>(11), hwWeight<SW>(12), hwWeight<SW>(13), hwWeight<SW>(14),     \
                hwWeight<SW>(15)};                                                          \
            float wx = kw[s];                                                               \
            bool left = 2 * s + 1 < SW;                                                     \
            float aa = left ? t[0] : t[1];                                                  \
            float bb = left ? t[1] : t[2];                                                  \
            float val = (1.f - wx) * aa + wx * bb;                                          \
            int kk = r * SW + s;                                                            \
            if (val > best[kk]) { best[kk] = val; bi[kk] = (c); }                           \
        }                                                                                   \
    }

// per-thread setup shared by ALG 9 / 10: thread -> (n, input row pp, sub-tile, column q)
#define FH_HW_SETUP                                                                         \
    constexpr int NSUB = SH / RT;                                                           \
    constexpr int NOUT = RT * SW;                                                           \
    int q = static_cast<int>(i % p.Win);                                                    \
    int64_t t_ = i / p.Win;                                                                 \
    int sub = static_cast<int>(t_ % NSUB);                                                  \
    t_ /= NSUB;                                                                             \
    int pp = static_cast<int>(t_ % p.Hin);                                                  \
    int n = static_cast<int>(t_ / p.Hin);                                                   \
    int rBase = sub * RT;                                                                   \
    int64_t ry[3] = {static_cast<int64_t>(max(pp - 1, 0)) * p.rowStride,                   \
        static_cast<int64_t>(pp) * p.rowStride,                                             \
        static_cast<int64_t>(min(pp + 1, p.Hin - 1)) * p.rowStride};                        \
    int64_t cx[3] = {static_cast<int64_t>(max(q - 1, 0)) * p.colStride,                     \
        static_cast<int64_t>(q) * p.colStride,                                              \
        static_cast<int64_t>(min(q + 1, p.Win - 1)) * p.colStride};                         \
    float wyv[RT];                                                                          \
    bool upv[RT];                                                                           \
    _Pragma("unroll") for (int r = 0; r < RT; ++r) hwRow<SH>(rBase + r, wyv[r], upv[r]);    \
    float best[NOUT];                                                                       \
    int bi[NOUT];                                                                           \
    _Pragma("unroll") for (int k = 0; k < NOUT; ++k) { best[k] = FH_NEG_INF; bi[k] = 0; }  \
    int64_t obase = (static_cast<int64_t>(n) * p.Hout + static_cast<int64_t>(SH) * pp + rBase) * p.Wout \
        + static_cast<int64_t>(SW) * q;

// ALG 9: any type and layout, one channel at a time from global memory
template <typename T, int BLOCK, int SH, int SW, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockHW(T const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int64_t total = static_cast<int64_t>(p.N) * p.Hin * (SH / RT) * p.Win;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        FH_HW_SETUP
        T const* base = in + n * p.batchStride;
        for (int c = 0; c < p.C; ++c)
        {
            T const* pl = base + fhChOff<GV>(c, p);   // ALG9 channel offset
            float v[3][3];
#pragma unroll
            for (int a = 0; a < 3; ++a)
#pragma unroll
                for (int k = 0; k < 3; ++k) v[a][k] = Cvt<T>::load(pl + ry[a] + cx[k], p.dq);
            FH_HW_CHANNEL(c, v)
        }
        fhStoreTileW<SW, RT>(out, obase, p.Wout, bi);
    }
}

// ALG 10: fp16 with 8 adjacent channels (vec8Layout; GV 0 or 32), 16-byte channel loads
template <int BLOCK, int SH, int SW, int RT, int GV = 0>
__global__ __launch_bounds__(BLOCK) void kBlockHWVecC(__half const* __restrict__ in, int32_t* __restrict__ out, BiParams p)
{
    int64_t total = static_cast<int64_t>(p.N) * p.Hin * (SH / RT) * p.Win;
    int64_t stride = static_cast<int64_t>(gridDim.x) * BLOCK;
    for (int64_t i = blockIdx.x * static_cast<int64_t>(BLOCK) + threadIdx.x; i < total; i += stride)
    {
        FH_HW_SETUP
        __half const* base = in + n * p.batchStride;
        int cb = 0;
        for (; cb + 8 <= p.C; cb += 8)
        {
            float vv[3][3][8];
#pragma unroll
            for (int a = 0; a < 3; ++a)
#pragma unroll
                for (int k = 0; k < 3; ++k) fhLoadH8(base + fhChOff<GV>(cb, p) + ry[a] + cx[k], vv[a][k]);
#pragma unroll
            for (int j = 0; j < 8; ++j)
            {
                float v[3][3];
#pragma unroll
                for (int a = 0; a < 3; ++a)
#pragma unroll
                    for (int k = 0; k < 3; ++k) v[a][k] = vv[a][k][j];
                FH_HW_CHANNEL(cb + j, v)
            }
        }
        for (int c = cb; c < p.C; ++c)
        {
            float v[3][3];
#pragma unroll
            for (int a = 0; a < 3; ++a)
#pragma unroll
                for (int k = 0; k < 3; ++k) v[a][k] = __half2float(__ldg(base + fhChOff<GV>(c, p) + ry[a] + cx[k]));
            FH_HW_CHANNEL(c, v)
        }
        fhStoreTileW<SW, RT>(out, obase, p.Wout, bi);
    }
}

// (SH, SW) pairs ALG 9 / 10 are built for: SH, SW in {1,2,3,4,6,8,16}, minus what the square
// kernels cover ({2,4,8} squared on the diagonal) and the identity 1 x 1
__host__ __device__ constexpr bool hwScaleOk(int sh, int sw)
{
    auto in = [](int s) { return s == 1 || s == 2 || s == 3 || s == 4 || s == 6 || s == 8 || s == 16; };
    bool squareFast = sh == sw && (sh == 2 || sh == 4 || sh == 8);
    return in(sh) && in(sw) && !squareFast && !(sh == 1 && sw == 1);
}

// Row tiles for ALG 9 / 10: divisors of SH with RT * SW <= 16 (accumulator registers)
__host__ __device__ constexpr bool hwRtOk(int sh, int sw, int rt)
{
    return rt >= 1 && rt <= 8 && sh % rt == 0 && rt * sw <= 16;
}

#undef FH_HW_SETUP
#undef FH_HW_CHANNEL

}  // namespace fused_head
