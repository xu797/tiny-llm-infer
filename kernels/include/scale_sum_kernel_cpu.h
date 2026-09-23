#ifndef MYVLLM_KERNELS_SCALE_SUM_H_
#define MYVLLM_KERNELS_SCALE_SUM_H_

#include "tensor.h"

namespace my_vllm 
{
void scale_sum_kernel_cpu(const Tensor& value, const Tensor& scale, 
                          const Tensor& output, int t, int d, int stride,
                          void* stream = nullptr);
}
#endif  
