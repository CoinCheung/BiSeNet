
#include <iostream>
#include <functional>
#include <algorithm>
#include <cfloat>
#include <thrust/pair.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include "NvInfer.h"



#define BLOCKSIZE 512

#define ivpair thrust::pair<scalar_t, int>


template<typename scalar_t>
__forceinline__ __device__ void reduce_max(ivpair* sdata, int blocksize, int tid) {
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
__global__ void arg_max_depth(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int *oten) {
    extern __shared__ __align__(sizeof(ivpair)) unsigned char sdata_raw[];
    ivpair *sdata = reinterpret_cast<ivpair*>(sdata_raw);
    sdata = sdata + blockDim.x * threadIdx.y;

    int sample_offset = gridDim.x * blockDim.y;
    int bid = threadIdx.y + blockIdx.x * blockDim.y;
    int samplesize = n_size * m_size;

    for (int i{bid}; i < samplesize; i += sample_offset) {
        int n_idx = i / m_size;
        int m_idx = i % m_size;

        /// NOTE: This is not memory-safe when dimsize < blockDim.x
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
        reduce_max(sdata, blockDim.x, threadIdx.x);

        idx = n_idx * m_size + m_idx;
        oten[idx] = sdata[0].second;
    }
}


template<typename scalar_t>
__global__ void arg_max_spatial(const int n_size,
                            const int dimsize, const int m_size,
                            const scalar_t *inten,
                            int *oten) {

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


void argMaxFunc(const void *inten,
                void *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream) {
    if (inten == nullptr or oten == nullptr) std::abort();

    int samplesize = n_size * m_size;
    int shm_size = 0;
    dim3 grid, block;

    if (dimsize <= 256) {
        int blockx, gridx;
        cudaOccupancyMaxPotentialBlockSize(&gridx, &blockx,
                arg_max_spatial<float>, 0, samplesize);
        gridx = std::min(4096, gridx << 2);
        block.x = blockx; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_spatial<float><<<grid, block, shm_size>>>(
                    n_size, dimsize, m_size,
                    reinterpret_cast<const float*>(inten),
                    reinterpret_cast<int*>(oten));
        } else {
            arg_max_spatial<float><<<grid, block, shm_size, *stream>>>(
                    n_size, dimsize, m_size,
                    reinterpret_cast<const float*>(inten),
                    reinterpret_cast<int*>(oten));
        }

    } else {
        int blockx, blocky, gridx;
        shm_size = (sizeof(float) + sizeof(int)) * BLOCKSIZE;
        int block_lmt = std::min(BLOCKSIZE, dimsize);
        blockx = 32;
        while (blockx <= block_lmt) blockx = (blockx << 1);
        blockx = (blockx >> 1); // must make sure dimsize > blockx
        blocky = BLOCKSIZE / blockx;
        gridx = std::min(4096, samplesize / blocky);
        block.x = blockx; block.y = blocky; grid.x = gridx;

        if (stream == nullptr) {
            arg_max_depth<float><<<grid, block, shm_size>>>(
                    n_size, dimsize, m_size,
                    reinterpret_cast<const float*>(inten),
                    reinterpret_cast<int*>(oten));
        } else {
            arg_max_depth<float><<<grid, block, shm_size, *stream>>>(
                    n_size, dimsize, m_size,
                    reinterpret_cast<const float*>(inten),
                    reinterpret_cast<int*>(oten));
        }
    }


}

