#include <cub/block/block_reduce.cuh>
#include <cuda_runtime_api.h>
#include <cfloat>
#include <cmath>
#include <cuda_fp16.h>

#include "paged_attention.h"
#include "cuda_alloc.h"

namespace my_vllm
{
namespace
{
constexpr int32_t kThreads = 256;

__device__ void paged_softmax(float* values, int32_t size)
{
    using BlockReduce = cub::BlockReduce<float, kThreads>;
    __shared__ typename BlockReduce::TempStorage storage;
    __shared__ float shared_value;
    const int32_t tid = threadIdx.x;
    float max_value = tid < size ? values[tid] : -FLT_MAX;
    for (int32_t index = tid + blockDim.x; index < size; index += blockDim.x)
    {
        max_value = fmaxf(max_value, values[index]);
    }
    max_value = BlockReduce(storage).Reduce(max_value, cub::Max());
    if (tid == 0) shared_value = max_value;
    __syncthreads();
    max_value = shared_value;

    float sum = 0.0f;
    for (int32_t index = tid; index < size; index += blockDim.x)
    {
        values[index] = expf(values[index] - max_value);
        sum += values[index];
    }
    sum = BlockReduce(storage).Sum(sum);
    if (tid == 0) shared_value = sum;
    __syncthreads();
    sum = shared_value;
    for (int32_t index = tid; index < size; index += blockDim.x)
    {
        values[index] /= sum;
    }
}

__global__ void paged_kv_cache_store_batch_kernel(
    int32_t rows, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t kv_dim, int32_t max_table_entries, const int32_t* positions,
    const int32_t* block_tables, const __half* keys, const __half* values,
    __half* key_cache, __half* value_cache)
{
    const int32_t index = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t count = rows * kv_dim;
    if (index >= count) return;

    const int32_t row = index / kv_dim;
    const int32_t dim = index % kv_dim;
    const int32_t position = positions[row];
    const int32_t page = block_tables[
        static_cast<size_t>(row) * max_table_entries + position / block_size];
    const size_t offset =
        ((static_cast<size_t>(layer_index) * num_blocks + page) * block_size +
         position % block_size) * kv_dim + dim;
    key_cache[offset] = keys[index];
    value_cache[offset] = values[index];
}

__global__ void paged_attention_batch_kernel(
    int32_t head_num, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t score_stride, int32_t max_table_entries, int32_t kv_dim,
    int32_t kv_mul, int32_t head_size, const int32_t* positions,
    const __half* query, __half* output, float* scores, const __half* key_cache,
    const __half* value_cache, const int32_t* block_tables)
{
    const int32_t row = blockIdx.y;
    const int32_t head = blockIdx.x;
    const int32_t position = positions[row];
    const int32_t context_len = position + 1;
    const int32_t head_offset = (head / kv_mul) * head_size;
    const int32_t query_offset = row * head_num * head_size + head * head_size;
    const int32_t table_offset = row * max_table_entries;
    const size_t score_offset =
        (static_cast<size_t>(row) * head_num + head) * score_stride;
    float* head_scores = scores + score_offset;
    const float scale = rsqrtf(static_cast<float>(head_size));

    for (int32_t token = threadIdx.x; token < context_len; token += blockDim.x)
    {
        const int32_t page = block_tables[table_offset + token / block_size];
        const size_t cache_offset =
            ((static_cast<size_t>(layer_index) * num_blocks + page) * block_size +
             token % block_size) * kv_dim + head_offset;
        float dot = 0.0f;
        for (int32_t dim = 0; dim < head_size; ++dim)
            dot += __half2float(query[query_offset + dim]) *
                   __half2float(key_cache[cache_offset + dim]);
        head_scores[token] = dot * scale;
    }
    __syncthreads();

    paged_softmax(head_scores, context_len);
    __syncthreads();

    const int32_t output_offset = row * head_num * head_size + head * head_size;
    for (int32_t dim = threadIdx.x; dim < head_size; dim += blockDim.x)
    {
        float value = 0.0f;
        for (int32_t token = 0; token < context_len; ++token)
        {
            const int32_t page = block_tables[table_offset + token / block_size];
            const size_t cache_offset =
                ((static_cast<size_t>(layer_index) * num_blocks + page) * block_size +
                 token % block_size) * kv_dim + head_offset;
            value += head_scores[token] * __half2float(value_cache[cache_offset + dim]);
        }
        output[output_offset + dim] = __float2half(value);
    }
}

__global__ void gather_rows_kernel(int32_t row_width, int32_t selected_rows,
                                   const int32_t* row_indices,
                                   const uint16_t* source, uint16_t* output)
{
    const int32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const int32_t count = selected_rows * row_width;
    if (index >= count) return;
    const int32_t selected_row = index / row_width;
    const int32_t column = index % row_width;
    output[index] = source[static_cast<size_t>(row_indices[selected_row]) * row_width + column];
}

}

void gather_rows_cuda(int32_t row_width, const std::vector<int32_t>& row_indices,
                      const Tensor& source, const Tensor& output, CudaConfig* config)
{
    CHECK(config != nullptr);
    CHECK_GT(row_width, 0);
    CHECK(!row_indices.empty());
    Tensor device_indices(DataType::kDataTypeInt32,
                          static_cast<int32_t>(row_indices.size()), true,
                          CUDADeviceAllocatorFactory::get_instance());
    const cudaError_t copy_error = cudaMemcpyAsync(
        device_indices.ptr<int32_t>(), row_indices.data(),
        row_indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice, config->stream);
    CHECK_EQ(copy_error, cudaSuccess);
    constexpr int32_t threads = 256;
    const int32_t count = static_cast<int32_t>(row_indices.size()) * row_width;
    gather_rows_kernel<<<(count + threads - 1) / threads, threads, 0, config->stream>>>(
        row_width, static_cast<int32_t>(row_indices.size()), device_indices.ptr<int32_t>(),
        source.ptr<uint16_t>(), const_cast<uint16_t*>(output.ptr<uint16_t>()));
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void paged_kv_cache_store_batch_cuda(
    int32_t rows, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t kv_dim, int32_t max_table_entries, const Tensor& positions,
    const Tensor& block_tables, const Tensor& keys, const Tensor& values,
    const Tensor& key_cache, const Tensor& value_cache, CudaConfig* config)
{
    CHECK(config != nullptr);
    CHECK_GT(rows, 0);
    const int32_t count = rows * kv_dim;
    constexpr int32_t threads = 256;
    const int32_t blocks = (count + threads - 1) / threads;
    paged_kv_cache_store_batch_kernel<<<blocks, threads, 0, config->stream>>>(
        rows, layer_index, num_blocks, block_size, kv_dim, max_table_entries,
        positions.ptr<int32_t>(), block_tables.ptr<int32_t>(),
        reinterpret_cast<const __half*>(keys.ptr<uint16_t>()),
        reinterpret_cast<const __half*>(values.ptr<uint16_t>()),
        reinterpret_cast<__half*>(const_cast<uint16_t*>(key_cache.ptr<uint16_t>())),
        reinterpret_cast<__half*>(const_cast<uint16_t*>(value_cache.ptr<uint16_t>())));
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

void paged_attention_batch_cuda(
    int32_t rows, int32_t head_num, int32_t layer_index, int32_t num_blocks,
    int32_t block_size, int32_t score_stride, int32_t max_table_entries,
    int32_t kv_dim, int32_t kv_mul, int32_t head_size, const Tensor& positions,
    const Tensor& output, const Tensor& query, const Tensor& score,
    const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& block_tables, CudaConfig* config)
{
    CHECK(config != nullptr);
    CHECK_GT(rows, 0);
    const dim3 grid(head_num, rows);
    paged_attention_batch_kernel<<<grid, kThreads, 0, config->stream>>>(
        head_num, layer_index, num_blocks, block_size, score_stride,
        max_table_entries, kv_dim, kv_mul, head_size, positions.ptr<int32_t>(),
        reinterpret_cast<const __half*>(query.ptr<uint16_t>()),
        reinterpret_cast<__half*>(const_cast<uint16_t*>(output.ptr<uint16_t>())),
        const_cast<float*>(score.ptr<float>()),
        reinterpret_cast<const __half*>(key_cache.ptr<uint16_t>()),
        reinterpret_cast<const __half*>(value_cache.ptr<uint16_t>()), block_tables.ptr<int32_t>());
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

}
