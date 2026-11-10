#include "add_kernel_cuda.cuh"
#include <cuda_fp16.h>

namespace my_vllm 
{

__global__ void add_kernel_cu_fp16(int32_t size, const __half* in1, const __half* in2, __half* out) {
    int32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid >= size)
    {
        return;
    }
    out[tid] = __hadd(in1[tid], in2[tid]);
}

void add_kernel_cu(const Tensor& input1, const Tensor& input2,
                   const Tensor& output, void* stream) 
{
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);
    int32_t size = static_cast<int32_t>(input1.size());
    CHECK_EQ(size, input2.size());
    CHECK_EQ(size, output.size());
    int32_t thread_num = 512;
    int32_t block_num = (size + thread_num - 1) / thread_num;
    if (stream) 
    {
        cudaStream_t stream_ = static_cast<CUstream_st*>(stream);
        add_kernel_cu_fp16<<<block_num, thread_num, 0, stream_>>>(size,
            reinterpret_cast<const __half*>(input1.ptr<uint16_t>()),
            reinterpret_cast<const __half*>(input2.ptr<uint16_t>()),
            reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    } 
    else 
    {
        add_kernel_cu_fp16<<<block_num, thread_num>>>(size,
            reinterpret_cast<const __half*>(input1.ptr<uint16_t>()),
            reinterpret_cast<const __half*>(input2.ptr<uint16_t>()),
            reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())));
    }
}
}  
