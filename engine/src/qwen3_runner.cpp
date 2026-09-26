#include "qwen3_runner.h"

namespace my_vllm::engine
{
Status Qwen3ModelRunner::configure_kv_cache(int32_t num_blocks, int32_t block_size,
                                            int32_t max_batch_tokens)
{
    return model_.configure_paged_kv_cache(num_blocks, block_size, max_batch_tokens);
}

Status Qwen3ModelRunner::run_batch(const std::vector<ModelBatchToken>& tokens,
                                   const std::vector<size_t>& sample_rows,
                                   std::vector<std::vector<float>>& logits)
{
    std::vector<int32_t> token_ids;
    std::vector<int32_t> positions;
    std::vector<std::vector<int32_t>> block_tables;
    token_ids.reserve(tokens.size());
    positions.reserve(tokens.size());
    block_tables.reserve(tokens.size());
    for (const ModelBatchToken& token : tokens)
    {
        token_ids.push_back(token.token_id);
        positions.push_back(token.position);
        block_tables.push_back(token.block_table);
    }
    return model_.forward_paged_batch(token_ids, positions, block_tables, sample_rows, logits);
}

Status Qwen3ModelRunner::tokenize(const std::string& prompt,
                                  std::vector<int32_t>& token_ids) const
{
    token_ids = model_.tokenize_prompt(prompt);
    if (token_ids.empty()) return InvalidArgument("The prompt encoded to no tokens.");
    return Success();
}

std::string Qwen3ModelRunner::decode(const std::vector<int32_t>& token_ids) const
{
    return model_.decode_tokens(token_ids);
}

bool Qwen3ModelRunner::is_eos(int32_t token_id) const
{
    return model_.is_sentence_ending(token_id);
}
}
