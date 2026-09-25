#ifndef MYVLLM_KERNELS_PAGED_ATTENTION_H_
#define MYVLLM_KERNELS_PAGED_ATTENTION_H_

#include "cuda_config.h"
#include "tensor.h"

namespace my_vllm
{
void paged_attention_cpu(int32_t position, int32_t head_num, int32_t layer_index,
                         int32_t num_blocks, int32_t block_size, int32_t score_stride,
                         int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                         const Tensor& output, const Tensor& query,
                         const Tensor& score, const Tensor& key_cache,
                         const Tensor& value_cache, const Tensor& block_table);

void paged_attention_cuda(int32_t position, int32_t head_num, int32_t layer_index,
                          int32_t num_blocks, int32_t block_size, int32_t score_stride,
                          int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                          const Tensor& output, const Tensor& query,
                          const Tensor& score, const Tensor& key_cache,
                          const Tensor& value_cache, const Tensor& block_table,
                          CudaConfig* config);
void paged_kv_cache_store_batch_cuda(
    int32_t rows, int32_t layer_index, int32_t num_blocks, int32_t block_size,
    int32_t kv_dim, int32_t max_table_entries, const Tensor& positions,
    const Tensor& block_tables, const Tensor& keys, const Tensor& values,
    const Tensor& key_cache, const Tensor& value_cache, CudaConfig* config);

void paged_attention_batch_cuda(
    int32_t rows, int32_t head_num, int32_t layer_index, int32_t num_blocks,
    int32_t block_size, int32_t score_stride, int32_t max_table_entries,
    int32_t kv_dim, int32_t kv_mul, int32_t head_size, const Tensor& positions,
    const Tensor& output, const Tensor& query, const Tensor& score,
    const Tensor& key_cache, const Tensor& value_cache,
    const Tensor& block_tables, CudaConfig* config);
}

#endif
