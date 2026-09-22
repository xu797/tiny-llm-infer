#ifndef MYVLLM_KERNELS_EMBED_CUDA_H_
#define MYVLLM_KERNELS_EMBED_CUDA_H_

#include "tensor.h"

namespace my_vllm
{
void emb_kernel_cu(const Tensor& input, const Tensor& weight,
                   const Tensor& output, int32_t vocab_size, void* stream = nullptr);
}
#endif  
