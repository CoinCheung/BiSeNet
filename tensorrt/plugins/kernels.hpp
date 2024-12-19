
#ifndef _KERNELS_HPP_
#define _KERNELS_HPP_

#include <iostream>
#include <functional>
#include <algorithm>
#include <cfloat>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "NvInfer.h"



#define BLOCKSIZE 256

#define ivpair thrust::pair<scalar_t, int>


template<typename scalar_t>
__forceinline__ __device__ void reduce_max_shm(ivpair* sdata, int blocksize, int tid) {
    __syncthreads();
    for (int s{blocksize / 2}; s > 0; s >>= 1) {
        if (tid < s) {
            if (sdata[tid].first < sdata[tid + s].first) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
}


template<typename scalar_t>
__forceinline__ __device__
void broadcast_block_x(scalar_t& val, int src_id) {
    __shared__ scalar_t shm; 
    if (threadIdx.x == src_id) {
        shm = val;
    }
    __syncthreads();
    val = shm;
}


template<typename scalar_t>
__forceinline__ __device__
void reduce_max(scalar_t& val, bool broadcast) {
    /* this requires:
     * 1. warp layout is along x axis
     * 2. blockDim.x should be divisble by 32
     * 3. blockDim.x should be less or equal to 1024
     * 4. warpSize should be 32
     * 5. only thread with threadIdx.x == 0 obtains correct answer */

    __syncthreads();
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 16));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 8));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 4));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 2));
    val = std::max(val, __shfl_down_sync(0xffffffff, val, 1));

    __shared__ scalar_t shm[32];

    if (threadIdx.x % 32 == 0) {
        shm[threadIdx.x >> 5] = val;
    }
    __syncthreads();

    val = scalar_t(0.);

    /* from here actually only one warp work */
    if (threadIdx.x < (blockDim.x >> 5)) {
        val = shm[threadIdx.x];
    }

    if (threadIdx.x < 32) {
        val = std::max(val, __shfl_down_sync(0xffffffff, val, 16));
        val = std::max(val, __shfl_down_sync(0xffffffff, val, 8));
        val = std::max(val, __shfl_down_sync(0xffffffff, val, 4));
        val = std::max(val, __shfl_down_sync(0xffffffff, val, 2));
        val = std::max(val, __shfl_down_sync(0xffffffff, val, 1));
    }

    if (broadcast) {
        broadcast_block_x(val, 0);
    }
}



template<typename scalar_t>
__global__ void arg_max_depth(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int32_t *oten) {
    extern __shared__ __align__(sizeof(ivpair)) unsigned char sdata_raw[];
    ivpair *sdata = reinterpret_cast<ivpair*>(sdata_raw);
    sdata = sdata + blockDim.x * threadIdx.y;

    int sample_offset = gridDim.x * blockDim.y;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int samplesize = n_size * m_size;

    for (int i{bid}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        /// NOTE: This is not reliable when dimsize < blockDim.x
        int idx = n_idx * dimsize * m_size + threadIdx.x * m_size + m_idx;
        ivpair maxp = thrust::make_pair(inten[idx], threadIdx.x);
        int j = threadIdx.x + blockDim.x;
        for (; j < dimsize; j += blockDim.x) {
            idx += blockDim.x * m_size;
            scalar_t val = inten[idx];
            if (val > maxp.first) {
                maxp = thrust::make_pair(val, j);
            }
        }
        sdata[threadIdx.x] = maxp;
        __syncthreads();
        reduce_max_shm(sdata, blockDim.x, threadIdx.x);

        idx = n_idx * m_size + m_idx;
        oten[idx] = sdata[0].second;
    }
}


template<typename scalar_t>
__global__ void arg_max_spatial(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int32_t *oten) {

    int sample_offset = gridDim.x * blockDim.x;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int samplesize = n_size * m_size;

    for (int i{tid}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        // obtain max
        int idx = n_idx * dimsize * m_size + m_idx;
        scalar_t max_val = inten[idx];
        int res = 0;
        for (int j{1}; j < dimsize; ++j) {
            idx += m_size;
            scalar_t val = inten[idx];
            if (val > max_val) {
                max_val = val;
                res = j;
            }
        }
        idx = n_idx * m_size + m_idx;
        oten[idx] = res;
    }
}


template<typename scalar_t>
void argMaxFunc(const scalar_t *inten,
                int32_t *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream) {

    if (inten == nullptr or oten == nullptr) std::terminate();

    int samplesize = n_size * m_size;
    dim3 grid, block;

    if (dimsize <= 128) {
        int blockx, gridx;
        cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
                arg_max_spatial<scalar_t>, 0, samplesize);
        gridx = std::min(4096, gridx << 2);
        block.x = blockx; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_spatial<scalar_t><<<grid, block, 0>>>(
                    n_size, dimsize, m_size, inten, oten);
        } else {
            arg_max_spatial<scalar_t><<<grid, block, 0, *stream>>>(
                    n_size, dimsize, m_size, inten, oten);
        }

    } else {
        int blockx, blocky, gridx;
        int shm_size = (sizeof(scalar_t) + sizeof(int)) * BLOCKSIZE;
        int block_lmt = std::min(BLOCKSIZE, dimsize);
        blockx = 32;
        while (blockx <= block_lmt) blockx = (blockx << 1);
        blockx = (blockx >> 1); // must make sure dimsize > blockx
        blocky = BLOCKSIZE / blockx;
        gridx = std::min(4096, samplesize / blocky);
        block.x = blockx; block.y = blocky; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_depth<scalar_t><<<grid, block, shm_size>>>(
                    n_size, dimsize, m_size, inten, oten);
        } else {
            arg_max_depth<scalar_t><<<grid, block, shm_size, *stream>>>(
                    n_size, dimsize, m_size, inten, oten);
        }
    }
}

#endif
