#ifndef _KERNELS_HPP_
#define _KERNELS_HPP_

#include <cuda.h>
#include <cuda_runtime.h>


void argMaxFunc(const void *inten,
                void *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream);

#endif
