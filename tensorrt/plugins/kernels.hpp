
#ifndef _KERNELS_HPP_
#define _KERNELS_HPP_

#include <iostream>
#include <functional>
#include <algorithm>
#include <cfloat>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include "NvInfer.h"



#define BLOCKSIZE 256



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
scalar_t shfl_down_sync_func(scalar_t val, uint32_t delta) {
    return __shfl_down_sync(0xffffffff, val, delta);
}

template<>
__forceinline__ __device__
int8_t shfl_down_sync_func(int8_t val, uint32_t delta) {
    int32_t ival = static_cast<int32_t>(val);
    ival = __shfl_down_sync(0xffffffff, ival, delta);
    return static_cast<int8_t>(ival);
}

template<typename scalar_t>
__forceinline__ __device__
scalar_t max_pair_shfl_func(scalar_t& val, int32_t& ind, const uint32_t delta) {
    scalar_t other_v = shfl_down_sync_func(val, delta);
    int32_t other_i = shfl_down_sync_func(ind, delta);

    if (other_v > val) {
        val = other_v;
        ind = other_i;
    }
}


template<typename scalar_t>
__forceinline__ __device__
void reduce_max(scalar_t& val, int32_t& ind, bool broadcast) {
    /* this requires:
     * 1. warp layout is along x axis
     * 2. blockDim.x should be divisble by 32
     * 3. blockDim.x should be less or equal to 1024
     * 4. warpSize should be 32
     * 5. only thread with threadIdx.x == 0 obtains correct answer */

    __syncthreads();
    max_pair_shfl_func(val, ind, 16);
    max_pair_shfl_func(val, ind, 8);
    max_pair_shfl_func(val, ind, 4);
    max_pair_shfl_func(val, ind, 2);
    max_pair_shfl_func(val, ind, 1);

    __shared__ scalar_t shm_v[32];
    __shared__ int32_t  shm_i[32];

    if (threadIdx.x % 32 == 0) {
        shm_v[threadIdx.x >> 5] = val;
        shm_i[threadIdx.x >> 5] = ind;
    }
    __syncthreads();

    /* from here actually only one warp work */
    if (threadIdx.x < 32) {
        val = shm_v[0];
        ind = shm_i[0];
        int32_t n_warps = (blockDim.x >> 5);
        if (threadIdx.x < n_warps) {
            val = shm_v[threadIdx.x];
            ind = shm_i[threadIdx.x];
        }
        max_pair_shfl_func(val, ind, 16);
        max_pair_shfl_func(val, ind, 8);
        max_pair_shfl_func(val, ind, 4);
        max_pair_shfl_func(val, ind, 2);
        max_pair_shfl_func(val, ind, 1);
    }

    if (broadcast) {
        broadcast_block_x(val, 0);
        broadcast_block_x(ind, 0);
    }
}



template<typename scalar_t>
__global__ void arg_max_depth(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int32_t *oten) {

    scalar_t max_val;
    int32_t max_ind;

    int samplesize = n_size * m_size;

    for (int i=blockIdx.x; i < samplesize; i += gridDim.x) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        /// NOTE: This is not reliable when dimsize < blockDim.x
        int idx = n_idx * dimsize * m_size + threadIdx.x * m_size + m_idx;
        int j = threadIdx.x + blockDim.x;
        max_val = inten[idx];
        max_ind = threadIdx.x;
        for (; j < dimsize; j += blockDim.x) {
            idx += blockDim.x * m_size;
            scalar_t val = inten[idx];
            if (val > max_val) {
                max_val = val;
                max_ind = j;
            }
        }
        reduce_max(max_val, max_ind, false);

        if (threadIdx.x == 0) {
            idx = n_idx * m_size + m_idx;
            oten[idx] = max_ind;
        }
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
        int blockx, gridx;
        int block_lmt = std::min(BLOCKSIZE, dimsize);
        blockx = 32;
        while (blockx <= block_lmt) blockx = (blockx << 1);
        blockx = (blockx >> 1); // must make sure dimsize > blockx
        gridx = std::min(16384, samplesize);
        block.x = blockx; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_depth<scalar_t><<<grid, block, 0>>>(
                    n_size, dimsize, m_size, inten, oten);
        } else {
            arg_max_depth<scalar_t><<<grid, block, 0, *stream>>>(
                    n_size, dimsize, m_size, inten, oten);
        }
    }
}

#endif
