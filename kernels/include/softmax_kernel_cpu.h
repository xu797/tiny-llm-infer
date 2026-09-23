#ifndef MYVLLM_KERNELS_SOFTMAX_H_
#define MYVLLM_KERNELS_SOFTMAX_H_

#include "tensor.h"

namespace my_vllm 
{
void softmax_inplace_cpu(const Tensor& input, void* stream = nullptr);
}  
#endif  // MYVLLM_KERNELS_SOFTMAX_H_
