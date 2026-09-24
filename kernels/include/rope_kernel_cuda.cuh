#ifndef MYVLLM_KERNELS_ROPE_CUDA_H_
#define MYVLLM_KERNELS_ROPE_CUDA_H_

#include "tensor.h"

namespace my_vllm
{
void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                    const Tensor& input_k, const Tensor& input_pos,
                    const Tensor& sin_cache, const Tensor& cos_cache, void* stream);

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const Tensor& sin_cache,
                           const Tensor& cos_cache, cudaStream_t stream);

void qwen3_sin_cos_cache_calc_cu(int32_t head_size, int32_t max_seq_len, float rope_theta,
                                 const Tensor& sin_cache, const Tensor& cos_cache,
                                 cudaStream_t stream);

void qwen3_rope_kernel_cu(int32_t query_heads, int32_t kv_heads, int32_t head_size,
                          const Tensor& input_q, const Tensor& input_k,
                          const Tensor& input_pos, const Tensor& sin_cache,
                          const Tensor& cos_cache, void* stream);

}  
#endif  // MYVLLM_KERNELS_ROPE_CUDA_H_
