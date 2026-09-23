#ifndef MYVLLM_KERNELS_INTERFACE_H
#define MYVLLM_KERNELS_INTERFACE_H

#include "tensor.h"
#include "cuda_config.h"

namespace my_vllm
{
typedef void (*AddKernel)(const Tensor& input1, const Tensor& input2,
                          const Tensor& output, void* stream);
AddKernel get_add_kernel(DeviceType device_type);

typedef void (*MatmulKernel)(const Tensor& input, const Tensor& weight,
                             const Tensor& output, float scale, const CudaConfig* config);                          
MatmulKernel get_matmul_kernel(DeviceType device_type);

typedef void (*MatmulKernelQuant)(const Tensor& input, const Tensor& weight,
                                  const Tensor& output, int32_t group_size,
                                  const Tensor& scale, const CudaConfig* config);
MatmulKernelQuant get_matmul_kernel_quant8(DeviceType device_type);

typedef void (*EmbeddingKernel)(const Tensor& input, const Tensor& weight,
                                const Tensor& output, int32_t vocab_size, void* stream);
EmbeddingKernel get_emb_kernel(DeviceType device_type);


typedef void (*SwigluKernel)(const Tensor& input1, const Tensor& input2,
                             const Tensor& output, void* stream);

typedef void (*MHAKernel)(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const Tensor& mha_out, const Tensor& query_tensor,
                          const Tensor& score_tensor,
                          const Tensor& key_cache_tensor,
                          const Tensor& value_cache_tensor, DeviceType device_type,
                          CudaConfig*);

typedef void (*RMSNormKernel)(const Tensor& input, const Tensor& weight,
                              const Tensor& output, void* stream);

typedef void (*RoPEKernel)(int32_t dim, int32_t kv_dim, int32_t head_size,
                           const Tensor& input_q, const Tensor& input_k,
                           const Tensor& input_pos, const Tensor& sin_cache,
                           const Tensor& cos_cache, void* stream);

typedef void (*ScaleKernel)(float scale, const Tensor& input, void* stream);

typedef void (*SoftmaxInplaceKernel)(const Tensor& input, void* stream);

typedef void (*ScaleSumKernel)(const Tensor& value, const Tensor& scale,
                               const Tensor& output, int t, int size, int stride,
                               void* stream);

void softmax_inplace_cpu(const float* input_ptr, size_t size);

MHAKernel get_mha_kernel(DeviceType device_type);

RMSNormKernel get_rmsnorm_kernel(DeviceType device_type);

RoPEKernel get_rope_kernel(DeviceType device_type);

ScaleKernel get_scale_kernel(DeviceType device_type);

SoftmaxInplaceKernel get_softmax_kernel(DeviceType device_type);

SwigluKernel get_swiglu_kernel(DeviceType device_type, void* stream = nullptr);

ScaleSumKernel get_scale_sum_kernel(DeviceType device_type);

}

#endif