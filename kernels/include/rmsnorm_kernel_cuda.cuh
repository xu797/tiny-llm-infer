#ifndef MYVLLM_KERNELS_RMSNORM_CUDA_H_
#define MYVLLM_KERNELS_RMSNORM_CUDA_H_

#include "tensor.h"

namespace my_vllm 
{
void rmsnorm_kernel_cu(const Tensor& input, const Tensor& weight,
                       const Tensor& output, void* stream = nullptr);
}
#endif  // MYVLLM_KERNELS_RMSNORM_CUDA_H_
