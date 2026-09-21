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


}

#endif