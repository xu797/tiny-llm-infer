#include "kernels_interface.h"
#include "matmul_kernel_cuda.cuh"
#include <cub/block/block_reduce.cuh>
#include <cuda_fp16.h>

namespace my_vllm 
{
namespace
{
__global__ void add_bias_rows_kernel(__half* output, const __half* bias, int rows, int columns)
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int count = rows * columns;
    if (index < count) output[index] = __hadd(output[index], bias[index % columns]);
}

template <int THREADS>
__global__ void matmul_fp16_gemv_kernel(const __half* input, const __half* weight,
                                       __half* output, int input_dim,
                                       float scale)
{
    const int output_index = blockIdx.x;
    const int tid = threadIdx.x;
    const __half* weight_row = weight + static_cast<size_t>(output_index) * input_dim;
    float partial = 0.0f;
    for (int col = tid; col < input_dim; col += THREADS)
        partial += __half2float(input[col]) * __half2float(weight_row[col]);
    using BlockReduce = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduce::TempStorage storage;
    const float sum = BlockReduce(storage).Sum(partial);
    if (tid == 0) output[output_index] = __float2half(sum * scale);
}

template <int THREADS>
__global__ void matmul_int8_kernel(const __half* input, const int8_t* weight,
                                   const float* scales, __half* output,
                                   int rows, int input_dim, int output_dim,
                                   int group_size, int groups_per_row)
{
    const int output_index = blockIdx.x;
    const int batch_row = blockIdx.y;
    const int tid = threadIdx.x;
    const int8_t* weight_row = weight + static_cast<size_t>(output_index) * input_dim;
    const float* scale_row = scales + static_cast<size_t>(output_index) * groups_per_row;
    float partial = 0.0f;
    for (int col = tid; col < input_dim; col += THREADS)
        partial += __half2float(input[static_cast<size_t>(batch_row) * input_dim + col]) *
                   static_cast<float>(weight_row[col]) * scale_row[col / group_size];
    using BlockReduce = cub::BlockReduce<float, THREADS>;
    __shared__ typename BlockReduce::TempStorage storage;
    const float sum = BlockReduce(storage).Sum(partial);
    if (tid == 0)
        output[static_cast<size_t>(batch_row) * output_dim + output_index] = __float2half(sum);
}
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
    uint16_t* output_ptr = const_cast<uint16_t*>(output.ptr<uint16_t>());

    const int rows = input.dims_size() == 1 ? 1 : input.get_dim(0);
    CHECK_EQ(input_dim, input.get_dim(input.dims_size() - 1));
    CHECK(config != nullptr && config->cublas_handle != nullptr);

    if (rows == 1)
    {
        constexpr int32_t threads = 128;
        matmul_fp16_gemv_kernel<threads><<<output_dim, threads, 0, config->stream>>>(
            reinterpret_cast<const __half*>(input.ptr<uint16_t>()),
            reinterpret_cast<const __half*>(weight.ptr<uint16_t>()),
            reinterpret_cast<__half*>(output_ptr), input_dim, scale);
        CHECK_EQ(cudaGetLastError(), cudaSuccess);
        return;
    }

    // Row-major X[B, K] * W[N, K]^T is column-major W^T[N, K] * X^T[K, B].
    const __half alpha = __float2half(scale);
    const __half beta = __float2half(0.0f);
    const cublasStatus_t status = cublasHgemm(
        config->cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, output_dim, rows, input_dim,
        &alpha, reinterpret_cast<const __half*>(weight.ptr<uint16_t>()), input_dim,
        reinterpret_cast<const __half*>(input.ptr<uint16_t>()), input_dim,
        &beta, reinterpret_cast<__half*>(output_ptr), output_dim);
    CHECK_EQ(status, CUBLAS_STATUS_SUCCESS);
}

void matmul_kernel_cu_int8(const Tensor& input, const Tensor& weight,
                           const Tensor& scales, const Tensor& output,
                           int32_t group_size, const CudaConfig* config)
{
    CHECK(config != nullptr && config->stream != nullptr);
    CHECK_GT(group_size, 0);
    const int32_t rows = input.dims_size() == 1 ? 1 : input.get_dim(0);
    const int32_t input_dim = input.get_dim(input.dims_size() - 1);
    const int32_t output_dim = weight.get_dim(0);
    const int32_t groups_per_row = (input_dim + group_size - 1) / group_size;
    CHECK_EQ(weight.get_dim(1), input_dim);
    CHECK_EQ(scales.size(), static_cast<size_t>(output_dim) * groups_per_row);
    constexpr int32_t threads = 128;
    const dim3 grid(output_dim, rows);
    matmul_int8_kernel<threads><<<grid, threads, 0, config->stream>>>(
        reinterpret_cast<const __half*>(input.ptr<uint16_t>()), weight.ptr<int8_t>(), scales.ptr<float>(),
        reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())), rows, input_dim, output_dim,
        group_size, groups_per_row);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void add_bias_rows_cuda(const Tensor& output, const Tensor& bias,
                        int32_t rows, int32_t columns, const CudaConfig* config)
{
    CHECK(config != nullptr && config->stream != nullptr);
    constexpr int32_t threads = 256;
    const int32_t count = rows * columns;
    add_bias_rows_kernel<<<(count + threads - 1) / threads, threads, 0, config->stream>>>(
        reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        reinterpret_cast<const __half*>(bias.ptr<uint16_t>()), rows, columns);
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

}
