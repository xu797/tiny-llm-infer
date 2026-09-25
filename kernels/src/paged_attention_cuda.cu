#include <cub/block/block_reduce.cuh>
#include <cuda_runtime_api.h>
#include <cfloat>
#include <cmath>

#include "paged_attention.h"

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

__global__ void paged_attention_kernel(
    int32_t position, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t score_stride, int32_t kv_dim, int32_t kv_mul, int32_t head_size,
    const float* query, float* output, float* scores, const float* key_cache,
    const float* value_cache, const int32_t* block_table)
{
    const int32_t head = blockIdx.x;
    const int32_t head_offset = (head / kv_mul) * head_size;
    const int32_t context_len = position + 1;
    const float scale = rsqrtf(static_cast<float>(head_size));
    float* head_scores = scores + head * score_stride;

    for (int32_t token = threadIdx.x; token < context_len; token += blockDim.x)
    {
        const int32_t page = block_table[token / block_size];
        const size_t cache_offset =
            ((static_cast<size_t>(layer_index) * num_blocks + page) * block_size +
             token % block_size) * kv_dim + head_offset;
        float dot = 0.0f;
        for (int32_t dim = 0; dim < head_size; ++dim)
        {
            dot += query[head * head_size + dim] * key_cache[cache_offset + dim];
        }
        head_scores[token] = dot * scale;
    }
    __syncthreads();

    paged_softmax(head_scores, context_len);
    __syncthreads();

    float* head_output = output + head * head_size;
    for (int32_t dim = threadIdx.x; dim < head_size; dim += blockDim.x)
    {
        float value = 0.0f;
        for (int32_t token = 0; token < context_len; ++token)
        {
            const int32_t page = block_table[token / block_size];
            const size_t cache_offset =
                ((static_cast<size_t>(layer_index) * num_blocks + page) * block_size +
                 token % block_size) * kv_dim + head_offset;
            value += head_scores[token] * value_cache[cache_offset + dim];
        }
        head_output[dim] = value;
    }
}

__global__ void paged_kv_cache_store_batch_kernel(
    int32_t rows, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t kv_dim, int32_t max_table_entries, const int32_t* positions,
    const int32_t* block_tables, const float* keys, const float* values,
    float* key_cache, float* value_cache)
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
    const float* query, float* output, float* scores, const float* key_cache,
    const float* value_cache, const int32_t* block_tables)
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
            dot += query[query_offset + dim] * key_cache[cache_offset + dim];
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
            value += head_scores[token] * value_cache[cache_offset + dim];
        }
        output[output_offset + dim] = value;
    }
}
}

void paged_attention_cuda(int32_t position, int32_t head_num, int32_t layer_index,
                          int32_t num_blocks, int32_t block_size, int32_t score_stride,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const Tensor& output, const Tensor& query,
                          const Tensor& score, const Tensor& key_cache,
                          const Tensor& value_cache, const Tensor& block_table,
                          CudaConfig* config)
{
    CHECK(config != nullptr);
    paged_attention_kernel<<<head_num, kThreads, 0, config->stream>>>(
        position, layer_index, num_blocks, block_size, score_stride, kv_dim, kv_mul,
        head_size, query.ptr<float>(), const_cast<float*>(output.ptr<float>()),
        const_cast<float*>(score.ptr<float>()), key_cache.ptr<float>(),
        value_cache.ptr<float>(), block_table.ptr<int32_t>());
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
        positions.ptr<int32_t>(), block_tables.ptr<int32_t>(), keys.ptr<float>(),
        values.ptr<float>(), const_cast<float*>(key_cache.ptr<float>()),
        const_cast<float*>(value_cache.ptr<float>()));
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
        query.ptr<float>(), const_cast<float*>(output.ptr<float>()),
        const_cast<float*>(score.ptr<float>()), key_cache.ptr<float>(),
        value_cache.ptr<float>(), block_tables.ptr<int32_t>());
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
}

}
