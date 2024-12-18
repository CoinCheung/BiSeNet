
#include "kernels.h"
#include <cuda_runtime.h>


template void argMaxFunc<float>(const float *inten,
                float *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream);


template void argMaxFunc<__half>(const __half *inten,
                __half *oten, const int n_size,
                const int dimsize, const int m_size,
                cudaStream_t* stream);
