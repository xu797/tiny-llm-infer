#ifndef MYVLLM_KERNELS_ADD_CPU_H_
#define MYVLLM_KERNELS_ADD_CPU_H_

#include "tensor.h"

namespace my_vllm
{

void add_kernel_cpu(const Tensor& input1, const Tensor& input2,
                    const Tensor& output, void* stream = nullptr);
}

#endif