#include <cub/block/block_reduce.cuh>

#include "kernels_interface.h"
#include "matmul_kernel_cuda.cuh"

namespace my_vllm 
{
template <int THREAD_PER_BLOCK, int ROW_PER_BLOCK>
__global__ void matmul_kernel_cu_fp32(const float* input, const float* weight, float* output, int M, int K) 
{
    __shared__ float sdata[THREAD_PER_BLOCK];
    unsigned int tid = threadIdx.x;

    int start_row = blockIdx.x * ROW_PER_BLOCK;
    int end_row = start_row + ROW_PER_BLOCK;
    if (start_row >= K) 
    {
        return;
    }

    constexpr int pack_size = 4;
    const int pack_num = M / pack_size;
    const int pack_off = pack_size * pack_num;

#pragma unroll
  for (int p = start_row; p < end_row; ++p) 
  {
      sdata[tid] = 0;
      int row_offset = p * M;
      float4* input_float4_ptr = (float4*)input;
      float4* weight_float4_ptr = (float4*)(weight + row_offset);

#pragma unroll
      for (int i = tid; i < pack_num; i += blockDim.x)
      {
          float4 input_float4 = *(input_float4_ptr + i);
          float4 weight_float4 = *(weight_float4_ptr + i);
          float part_sum = input_float4.x * weight_float4.x + input_float4.y * weight_float4.y +
                          input_float4.z * weight_float4.z + input_float4.w * weight_float4.w;
          sdata[tid] += part_sum;
      }

      for (int i = pack_off + tid; i < M; i += blockDim.x) 
      {
          sdata[tid] += input[i] * weight[row_offset + i];
      }

      __syncthreads();

      using BlockReduce = cub::BlockReduce<float, THREAD_PER_BLOCK>;
      __shared__ typename BlockReduce::TempStorage temp;
      float part_sum = BlockReduce(temp).Sum(sdata[tid]);
      __syncthreads();

      if (tid == 0) 
      {
        output[p] = part_sum;
      }
      __syncthreads();
  }
}

template <int ROWS_PER_BLOCK, int COLS_PER_BLOCK>
__global__ void matmul_kernel_cu_batched(const float* input, const float* weight,
                                         float* output, int rows, int input_dim,
                                         int output_dim, float scale)
{
    const int row = blockIdx.y * ROWS_PER_BLOCK + threadIdx.y;
    const int col = blockIdx.x * COLS_PER_BLOCK + threadIdx.x;
    if (row >= rows || col >= output_dim) return;

    float sum = 0.0f;
    const float* input_row = input + static_cast<size_t>(row) * input_dim;
    const float* weight_row = weight + static_cast<size_t>(col) * input_dim;
    for (int index = 0; index < input_dim; ++index)
        sum += input_row[index] * weight_row[index];
    output[static_cast<size_t>(row) * output_dim + col] = sum * scale;
}

void matmul_kernel_cu(const Tensor& input, const Tensor& weight,
                      const Tensor& output, const float scale, const CudaConfig* config)
{
    CHECK(!input.is_empty() && input.dims_size() <= 2);
    CHECK(input.device_type() == DeviceType::kDeviceCUDA);
    CHECK(!weight.is_empty() && weight.dims_size() == 2);
    CHECK(weight.device_type() == DeviceType::kDeviceCUDA);
    const int32_t output_dim = weight.get_dim(0);
    const int32_t input_dim = weight.get_dim(1);
    float* output_ptr = const_cast<float*>(output.ptr<float>());

    if (input.dims_size() == 1)
    {
        CHECK_EQ(input_dim, input.get_dim(0));
        if (config && config->stream)
            matmul_kernel_cu_fp32<128, 1><<<output_dim, 128, 0, config->stream>>>(
                input.ptr<float>(), weight.ptr<float>(), output_ptr, input_dim, output_dim);
        else
            matmul_kernel_cu_fp32<128, 1><<<output_dim, 128>>>(
                input.ptr<float>(), weight.ptr<float>(), output_ptr, input_dim, output_dim);
    }
    else
    {
        CHECK_EQ(input_dim, input.get_dim(1));
        const int rows = input.get_dim(0);
        const dim3 block(32, 8);
        const dim3 grid((output_dim + 31) / 32, (rows + 7) / 8);
        if (config && config->stream)
            matmul_kernel_cu_batched<8, 32><<<grid, block, 0, config->stream>>>(
                input.ptr<float>(), weight.ptr<float>(), output_ptr, rows, input_dim,
                output_dim, scale);
        else
            matmul_kernel_cu_batched<8, 32><<<grid, block>>>(
                input.ptr<float>(), weight.ptr<float>(), output_ptr, rows, input_dim,
                output_dim, scale);
    }
}

}  