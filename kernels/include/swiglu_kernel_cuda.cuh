#ifndef MYVLLM_KERNELS_SWIGLU_CUDA_H_
#define MYVLLM_KERNELS_SWIGLU_CUDA_H_

#include "tensor.h"

namespace my_vllm
{
void swiglu_kernel_cu(const Tensor& input1, const Tensor& input2,
                      const Tensor& output, void* stream);
}
#endif  // MYVLLM_KERNELS_SWIGLU_CUDA_H_
