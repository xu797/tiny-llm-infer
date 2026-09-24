#include <cub/block/block_reduce.cuh>

#include "rmsnorm_kernel_cuda.cuh"

namespace my_vllm
{
template <int32_t BLOCK_DIM>
static __global__ void row_rmsnorm_f32(const float* in, const float* wei, float* out,
                                       int row_size, float eps)
{
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    in += row * row_size;
    out += row * row_size;

    float sum = 0.0f;
    for (int i = tid; i < row_size; i += blockDim.x)
    {
        sum += in[i] * in[i];
    }

    using BlockReduce = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduce::TempStorage temp;
    __shared__ float shared_val;
    sum = BlockReduce(temp).Sum(sum);
    if (threadIdx.x == 0) 
    {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(row_size) + eps);
    for (int i = tid; i < row_size; i += blockDim.x)
    {
        out[i] = wei[i] * in[i] * scale;
    }
}

void rmsnorm_kernel_cu(const Tensor& input, const Tensor& weight, const Tensor& output,
                       void* stream, float eps)
{
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == DeviceType::kDeviceCUDA &&
        weight.device_type() == DeviceType::kDeviceCUDA &&
        output.device_type() == DeviceType::kDeviceCUDA);

    const int32_t row_size = static_cast<int32_t>(weight.size());
    CHECK_GT(row_size, 0);
    CHECK_EQ(input.size() % row_size, 0);
    CHECK_EQ(output.size(), input.size());
    const int32_t row_num = static_cast<int32_t>(input.size() / row_size);
    const float* in_ptr = input.ptr<float>();
    float* wei_ptr = const_cast<float*>(weight.ptr<float>());
    float* out_ptr = const_cast<float*>(output.ptr<float>());
    constexpr int threads_num = 128;
    if (stream) 
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        row_rmsnorm_f32<128><<<row_num, threads_num, 0, stream_>>>(in_ptr, wei_ptr, out_ptr, row_size, eps);
    } 
    else
    {
        row_rmsnorm_f32<128><<<row_num, threads_num>>>(in_ptr, wei_ptr, out_ptr, row_size, eps);
    }
}
}  // namespace kernel
