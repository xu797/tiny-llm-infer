#include "sequence.h"

#include <algorithm>

namespace my_vllm::engine
{
Sequence::Sequence(uint64_t id, std::vector<int32_t> prompt_tokens,
                   SamplingParams sample, int32_t block_size)
    : sampling(sample),
      id_(id),
      token_ids_(std::move(prompt_tokens)),
      prompt_size_(token_ids_.size()),
      block_size_(block_size),
      rng_(sample.seed == 0 ? id : sample.seed)
{
}

std::vector<int32_t> Sequence::completion_tokens() const
{
    return std::vector<int32_t>(token_ids_.begin() + static_cast<std::ptrdiff_t>(prompt_size_),
                                token_ids_.end());
}

std::vector<int32_t> Sequence::block_tokens(size_t block_index) const
{
    const size_t begin = block_index * static_cast<size_t>(block_size_);
    const size_t end = std::min(begin + static_cast<size_t>(block_size_), token_ids_.size());
    if (begin >= end) return {};
    return std::vector<int32_t>(token_ids_.begin() + static_cast<std::ptrdiff_t>(begin),
                                token_ids_.begin() + static_cast<std::ptrdiff_t>(end));
}

size_t Sequence::num_blocks() const
{
    return (token_ids_.size() + static_cast<size_t>(block_size_) - 1) /
           static_cast<size_t>(block_size_);
}

size_t Sequence::last_block_tokens() const
{
    const size_t count = num_blocks();
    return token_ids_.size() - (count - 1) * static_cast<size_t>(block_size_);
}

void Sequence::append_token(int32_t token)
{
    token_ids_.push_back(token);
}
}
