#ifndef MYVLLM_KERNELS_ADD_CUDA_H_
#define MYVLLM_KERNELS_ADD_CUDA_H_

#include "tensor.h"

namespace my_vllm
{

void add_kernel_cu(const Tensor& input1, const Tensor& input2,
                    const Tensor& output, void* stream = nullptr);
}

#endif