#ifndef MYVLLM_ENGINE_SEQUENCE_H_
#define MYVLLM_ENGINE_SEQUENCE_H_

#include <cstdint>
#include <random>
#include <vector>

#include "sampling_params.h"

namespace my_vllm::engine
{
enum class SequenceStatus
{
    kWaiting,
    kRunning,
    kFinished,
};

class Sequence
{
public:
    Sequence(uint64_t id, std::vector<int32_t> prompt_tokens,
             SamplingParams sampling, int32_t block_size);

    uint64_t id() const { return id_; }
    size_t size() const { return token_ids_.size(); }
    size_t prompt_size() const { return prompt_size_; }
    size_t completion_size() const { return token_ids_.size() - prompt_size_; }
    int32_t last_token() const { return token_ids_.back(); }
    int32_t token_at(size_t index) const { return token_ids_.at(index); }
    const std::vector<int32_t>& token_ids() const { return token_ids_; }
    std::vector<int32_t> completion_tokens() const;
    std::vector<int32_t> block_tokens(size_t block_index) const;
    size_t num_blocks() const;
    size_t last_block_tokens() const;
    void append_token(int32_t token);

    SamplingParams sampling;
    SequenceStatus status = SequenceStatus::kWaiting;
    std::vector<int32_t> block_table;
    size_t num_cached_tokens = 0;
    size_t num_scheduled_tokens = 0;
    bool is_prefill = true;

    std::mt19937_64& rng() { return rng_; }
    int32_t block_size() const { return block_size_; }

private:
    uint64_t id_;
    std::vector<int32_t> token_ids_;
    size_t prompt_size_;
    int32_t block_size_;
    std::mt19937_64 rng_;
};
}

#endif
