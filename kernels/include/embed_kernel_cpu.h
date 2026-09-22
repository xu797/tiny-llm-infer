#ifndef MYVLLM_KERNELS_EMBED_CPU_H_
#define MYVLLM_KERNELS_EMBED_CPU_H_

#include "status.h"
#include "tensor.h"

namespace my_vllm 
{
void emb_kernel_normal(const Tensor& input, const Tensor& weight,
                       const Tensor& output, int32_t vocab_size,
                       void* stream = nullptr);
}  
#endif  // MYVLLM_KERNELS_EMBED_CPU_H_
