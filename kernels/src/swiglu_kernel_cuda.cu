#include "tensor.h"
#include "swiglu_kernel_cuda.cuh"
#include <cuda_fp16.h>

namespace my_vllm 
{
__global__ void swiglu_kernel_cu_fp16(int size, const __half* in1, const __half* in2, __half* out)
{
    int tid = threadIdx.x;
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) 
    {
        return;
    }
    extern __shared__ float shared_mem[];
    float* smem1 = shared_mem;
    float* smem2 = shared_mem + blockDim.x;

    smem1[tid] = __half2float(in1[idx]);
    smem2[tid] = __half2float(in2[idx]);
    __syncthreads();

    float value = 1.0f / (1.0f + exp(-smem1[tid]));
    smem1[tid] = smem1[tid] * value;

    out[idx] = __float2half(smem1[tid] * smem2[tid]);
}

void swiglu_kernel_cu(const Tensor& input1, const Tensor& input2, const Tensor& output, void* stream) 
{
    CHECK_EQ(input1.is_empty(), false);
    CHECK(input1.device_type() == DeviceType::kDeviceCUDA);

    CHECK_EQ(input2.is_empty(), false);
    CHECK(input2.device_type() == DeviceType::kDeviceCUDA);

    CHECK_EQ(output.is_empty(), false);
    CHECK(output.device_type() == DeviceType::kDeviceCUDA);

    int size = static_cast<int32_t>(input1.size());
    int threads = 128;
    int blocks = (size + threads - 1) / threads;
    const size_t shmem = threads * sizeof(float) * 2;
    if (!stream) 
    {
        swiglu_kernel_cu_fp16<<<blocks, threads, shmem>>>(size,
            reinterpret_cast<const __half*>(input1.ptr<uint16_t>()),
            reinterpret_cast<const __half*>(input2.ptr<uint16_t>()),
            reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    } 
    else 
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        swiglu_kernel_cu_fp16<<<blocks, threads, shmem, stream_>>>(size,
            reinterpret_cast<const __half*>(input1.ptr<uint16_t>()),
            reinterpret_cast<const __half*>(input2.ptr<uint16_t>()),
            reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
}
}
