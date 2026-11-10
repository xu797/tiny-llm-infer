#ifndef MYVLLM_ENGINE_MODEL_RUNNER_H_
#define MYVLLM_ENGINE_MODEL_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "status.h"
#include "sampling_params.h"

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
    virtual Status run_batch_sampled(const std::vector<ModelBatchToken>& tokens,
                                     const std::vector<size_t>& sample_rows,
                                     const std::vector<float>& temperatures,
                                     const std::vector<uint64_t>& seeds,
                                     std::vector<int32_t>& token_ids)
    {
        std::vector<std::vector<float>> logits;
        const Status status = run_batch(tokens, sample_rows, logits);
        if (!status) return status;
        if (logits.size() != temperatures.size() || logits.size() != seeds.size())
            return InvalidArgument("Sampling parameters do not match the logits batch.");
        token_ids.clear();
        token_ids.reserve(logits.size());
        for (size_t row_index = 0; row_index < logits.size(); ++row_index)
        {
            const auto& row = logits[row_index];
            if (row.empty()) return InternalError("The model runner returned empty logits.");
            if (temperatures[row_index] <= 1.0e-6f)
            {
                token_ids.push_back(static_cast<int32_t>(
                    std::distance(row.begin(), std::max_element(row.begin(), row.end()))));
                continue;
            }
            const float max_logit = *std::max_element(row.begin(), row.end());
            std::vector<double> weights(row.size());
            const double inverse_temperature = 1.0 / temperatures[row_index];
            for (size_t index = 0; index < row.size(); ++index)
                weights[index] = std::exp((static_cast<double>(row[index]) - max_logit) *
                                          inverse_temperature);
            std::mt19937_64 rng(seeds[row_index]);
            std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
            token_ids.push_back(static_cast<int32_t>(distribution(rng)));
        }
        return Success();
    }
    virtual Status tokenize(const std::string& prompt,
                            std::vector<int32_t>& token_ids) const = 0;
    virtual std::string decode(const std::vector<int32_t>& token_ids) const = 0;
    virtual bool is_eos(int32_t token_id) const = 0;
};
}

#endif
