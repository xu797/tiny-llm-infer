#ifndef MYVLLM_KERNELS_ROPE_CPU_H_
#define MYVLLM_KERNELS_ROPE_CPU_H_
#include "tensor.h"

namespace my_vllm 
{
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache);

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                     const Tensor& input_k, const Tensor& input_pos,
                     const Tensor& sin_cache, const Tensor& cos_cache,
                     void* stream);
}  
#endif  // MYVLLM_KERNELS_ROPE_CPU_H_
