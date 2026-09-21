#ifndef MYVLLM_UTILS_CUDACONFIG_H_
#define MYVLLM_UTILS_CUDACONFIG_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace my_vllm 
{
struct CudaConfig 
{
    cudaStream_t stream = nullptr;
    ~CudaConfig() 
    {
        if (stream) 
        {
            cudaStreamDestroy(stream);
        }
    }
};

}

#endif