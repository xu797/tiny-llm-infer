#ifndef MYVLLM_KERNELS_SCALE_H_
#define MYVLLM_KERNELS_SCALE_H_
#include "tensor.h"

namespace my_vllm 
{
void scale_inplace_cpu(float scale, const Tensor& tensor, void* stream = nullptr);
}
#endif  // MYVLLM_KERNELS_SCALE_H_
