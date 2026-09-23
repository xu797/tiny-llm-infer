#ifndef MYVLLM_KERNELS_SWIGLU_H_
#define MYVLLM_KERNELS_SWIGLU_H_

#include "tensor.h"

namespace my_vllm 
{
void swiglu_kernel_cpu(const Tensor& input1, const Tensor& input2,
                       const Tensor& output, void* stream);
} 
#endif  // MYVLLM_KERNELS_SWIGLU_H_
