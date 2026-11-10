#ifndef MYVLLM_UTILS_CUDACONFIG_H_
#define MYVLLM_UTILS_CUDACONFIG_H_

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

namespace my_vllm 
{
struct CudaConfig 
{
    cudaStream_t stream = nullptr;
    cublasHandle_t cublas_handle = nullptr;
    ~CudaConfig() 
    {
        if (cublas_handle)
        {
            cublasDestroy(cublas_handle);
        }
        if (stream) 
        {
            cudaStreamDestroy(stream);
        }
    }
};

}

#endif
