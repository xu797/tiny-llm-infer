#ifndef MYVLLM_KERNELS_MHA_CPU_H_
#define MYVLLM_KERNELS_MHA_CPU_H_

#include "cuda_config.h"
#include "status.h"
#include "tensor.h"

namespace my_vllm
{
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const Tensor& mha_out,
                const Tensor& query_tensor, const Tensor& score_tensor,
                const Tensor& key_cache_tensor, const Tensor& value_cache_tensor,
                DeviceType device_type, CudaConfig* config);
}

#endif