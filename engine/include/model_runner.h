#ifndef MYVLLM_ENGINE_MODEL_RUNNER_H_
#define MYVLLM_ENGINE_MODEL_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "status.h"

namespace my_vllm::engine
{
struct ModelBatchToken
{
    int32_t token_id = -1;
    int32_t position = 0;
    std::vector<int32_t> block_table;
};

class ModelRunner
{
public:
    virtual ~ModelRunner() = default;
    virtual Status configure_kv_cache(int32_t num_blocks, int32_t block_size,
                                      int32_t max_batch_tokens) = 0;
    virtual Status run_batch(const std::vector<ModelBatchToken>& tokens,
                             const std::vector<size_t>& sample_rows,
                             std::vector<std::vector<float>>& logits) = 0;
    virtual Status tokenize(const std::string& prompt,
                            std::vector<int32_t>& token_ids) const = 0;
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;
    virtual bool is_eos(int32_t token_id) const = 0;
};
}

#endif
