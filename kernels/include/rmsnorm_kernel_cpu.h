#ifndef MYVLLM_KERNELS_RMSNORM_H_
#define MYVLLM_KERNELS_RMSNORM_H_

#include "tensor.h"

namespace my_vllm 
{
void rmsnorm_kernel_cpu(const Tensor& input, const Tensor& weight,
                        const Tensor& output, void* stream = nullptr, float eps = 1e-5f);
}  
#endif  // MYVLLM_KERNELS_RMSNORM_H_
