#include "embed_kernel_cuda.cuh"

namespace my_vllm 
{
__global__ void emb_kernel_cu_fp32(int32_t vocab_size, int32_t token_num, int32_t weight_dim,
                                   const int32_t* input_ptr, const float* weight_ptr,
                                   float* output_ptr) 
{
    int32_t token_idx = blockIdx.x;
    if (token_idx >= token_num) 
    {
        return;
    }
    int32_t token = input_ptr[token_idx];
    if (token >= vocab_size) 
    {
        return;
    }

    float* output_ptr_start = output_ptr + token_idx * weight_dim;
    const float* weight_ptr_start = weight_ptr + token * weight_dim;

    for (int32_t i = threadIdx.x; i < weight_dim; i += blockDim.x) 
    {
        output_ptr_start[i] = weight_ptr_start[i];
    }
}

void emb_kernel_cu(const Tensor& input, const Tensor& weight,
                   const Tensor& output, int32_t vocab_size, void* stream) 
{
    Tensor input_cu = input;
    const bool copied_input = input.device_type() != DeviceType::kDeviceCUDA;
    if (input.device_type() != DeviceType::kDeviceCUDA) 
    {
        input_cu = input.clone();
        input_cu.to_cuda(nullptr);
    }
    const int32_t input_num = static_cast<int32_t>(input.size());
    const int32_t weight_dim = weight.get_dim(1);
    CHECK(weight.device_type() == output.device_type());
    CHECK(output.device_type() == DeviceType::kDeviceCUDA);

    constexpr int32_t thread_num = 128;
    int32_t* in_ptr = input_cu.ptr<int32_t>();
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    if (stream) 
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        emb_kernel_cu_fp32<<<input_num, thread_num, 0, stream_>>>(vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    } 
    else 
    {
        emb_kernel_cu_fp32<<<input_num, thread_num>>>(vocab_size, input_num, weight_dim, in_ptr, wei_ptr, out_ptr);
    }
    CHECK_EQ(cudaGetLastError(), cudaSuccess);

    // input_cu owns a temporary device copy for CPU token ids. Do not return it
    // to the allocator's cache while the embedding kernel is still reading it.
    if (copied_input)
    {
        cudaError_t status;
        if (stream)
        {
            status = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
        }
        else
        {
            status = cudaDeviceSynchronize();
        }
        CHECK_EQ(status, cudaSuccess);
    }
}
}
