#include "paged_attention.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace my_vllm
{
void paged_attention_cpu(int32_t position, int32_t head_num, int32_t layer_index,
                         int32_t num_blocks, int32_t block_size, int32_t score_stride,
                         int32_t kv_dim, int32_t kv_mul, int32_t head_size,
                         const Tensor& output, const Tensor& query,
                         const Tensor& score, const Tensor& key_cache,
                         const Tensor& value_cache, const Tensor& block_table)
{
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
    const int32_t context_len = position + 1;
    const float* query_ptr = query.ptr<float>();
    const int32_t* table = block_table.ptr<int32_t>();
    const float* keys = key_cache.ptr<float>();
    const float* values = value_cache.ptr<float>();
    float* scores = const_cast<float*>(score.ptr<float>());
    float* output_ptr = const_cast<float*>(output.ptr<float>());
    std::vector<float> local_scores(static_cast<size_t>(context_len));

    for (int32_t head = 0; head < head_num; ++head)
    {
        const int32_t kv_head_offset = (head / kv_mul) * head_size;
        for (int32_t token = 0; token < context_len; ++token)
        {
            const int32_t logical_block = token / block_size;
            const int32_t physical_block = table[logical_block];
            const size_t cache_offset =
                ((static_cast<size_t>(layer_index) * num_blocks + physical_block) * block_size +
                 token % block_size) * kv_dim + kv_head_offset;
            float dot = 0.0f;
            for (int32_t dim = 0; dim < head_size; ++dim)
            {
                dot += query_ptr[head * head_size + dim] * keys[cache_offset + dim];
            }
            local_scores[static_cast<size_t>(token)] = dot * scale;
        }

        const float max_score = *std::max_element(local_scores.begin(), local_scores.end());
        float denominator = 0.0f;
        for (float& value : local_scores)
        {
            value = std::exp(value - max_score);
            denominator += value;
        }
        float* head_scores = scores + head * score_stride;
        for (int32_t token = 0; token < context_len; ++token)
        {
            head_scores[token] = local_scores[static_cast<size_t>(token)] / denominator;
        }

        float* head_output = output_ptr + head * head_size;
        std::fill(head_output, head_output + head_size, 0.0f);
        for (int32_t token = 0; token < context_len; ++token)
        {
            const int32_t physical_block = table[token / block_size];
            const size_t cache_offset =
                ((static_cast<size_t>(layer_index) * num_blocks + physical_block) * block_size +
                 token % block_size) * kv_dim + kv_head_offset;
            const float weight = head_scores[token];
            for (int32_t dim = 0; dim < head_size; ++dim)
            {
                head_output[dim] += weight * values[cache_offset + dim];
            }
        }
    }
}
}
