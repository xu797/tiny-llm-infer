#ifndef MYVLLM_ENGINE_QWEN3_RUNNER_H_
#define MYVLLM_ENGINE_QWEN3_RUNNER_H_

#include "model_runner.h"
#include "qwen3.h"

namespace my_vllm::engine
{
class Qwen3ModelRunner final : public ModelRunner
{
public:
    explicit Qwen3ModelRunner(Qwen3Model& model) : model_(model) {}

    Status configure_kv_cache(int32_t num_blocks, int32_t block_size,
                              int32_t max_batch_tokens) override;
    Status run_batch(const std::vector<ModelBatchToken>& tokens,
                     const std::vector<size_t>& sample_rows,
                     std::vector<std::vector<float>>& logits) override;
    Status tokenize(const std::string& prompt,
                    std::vector<int32_t>& token_ids) const override;
    std::string decode(const std::vector<int32_t>& token_ids) const override;
    bool is_eos(int32_t token_id) const override;

private:
    Qwen3Model& model_;
};
}

#endif
