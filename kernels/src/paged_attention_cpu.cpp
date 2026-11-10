#include "paged_attention.h"

#include <algorithm>
#include <cmath>
#include "fp16.h"

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
    const uint16_t* query_ptr = query.ptr<uint16_t>();
    const int32_t* table = block_table.ptr<int32_t>();
    const uint16_t* keys = key_cache.ptr<uint16_t>();
    const uint16_t* values = value_cache.ptr<uint16_t>();
    float* scores = const_cast<float*>(score.ptr<float>());
    uint16_t* output_ptr = const_cast<uint16_t*>(output.ptr<uint16_t>());
    for (int32_t head = 0; head < head_num; ++head)
    {
        const int32_t kv_head_offset = (head / kv_mul) * head_size;
        float* head_scores = scores + head * score_stride;
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
                dot += fp16_to_float(query_ptr[head * head_size + dim]) *
                       fp16_to_float(keys[cache_offset + dim]);
            }
            head_scores[token] = dot * scale;
        }

        const float max_score = *std::max_element(head_scores, head_scores + context_len);
        float denominator = 0.0f;
        for (int32_t token = 0; token < context_len; ++token)
        {
            head_scores[token] = std::exp(head_scores[token] - max_score);
            denominator += head_scores[token];
        }
        for (int32_t token = 0; token < context_len; ++token)
        {
            head_scores[token] /= denominator;
        }

        uint16_t* head_output = output_ptr + head * head_size;
        for (int32_t dim = 0; dim < head_size; ++dim)
        {
            float value = 0.0f;
            for (int32_t token = 0; token < context_len; ++token)
            {
                const int32_t physical_block = table[token / block_size];
                const size_t cache_offset =
                    ((static_cast<size_t>(layer_index) * num_blocks + physical_block) * block_size +
                     token % block_size) * kv_dim + kv_head_offset;
                value += head_scores[token] * fp16_to_float(values[cache_offset + dim]);
            }
            head_output[dim] = float_to_fp16(value);
        }
    }
}
}
