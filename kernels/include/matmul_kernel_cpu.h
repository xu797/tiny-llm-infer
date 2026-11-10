#ifndef MYVLLM_KERNELS_MATMUL_CPU_H_
#define MYVLLM_KERNELS_MATMUL_CPU_H_

#include "tensor.h"
#include "cuda_config.h"

namespace my_vllm
{

void matmul_kernel_cpu(const Tensor& input, const Tensor& weight,
                    const Tensor& output, float scale = 1.f,
                    const CudaConfig* config = nullptr);
void matmul_kernel_cpu_int8(const Tensor& input, const Tensor& weight,
                            const Tensor& scales, const Tensor& output,
                            int32_t group_size);
                    
}

#endif
