#ifndef MYVLLM_KERNELS_MATMUL_CUDA_H_
#define MYVLLM_KERNELS_MATMUL_CUDA_H_

#include "tensor.h"
#include "cuda_config.h"

namespace my_vllm
{

void matmul_kernel_cu(const Tensor& input, const Tensor& weight,
                      const Tensor& output, float scale = 1.f,
                      const CudaConfig* config = nullptr);

void matmul_kernel_cu_qint8(const Tensor& input, const Tensor& weight,
                            const Tensor& output, int32_t group_size,
                            const Tensor& scale, const CudaConfig* config = nullptr);
}

#endif