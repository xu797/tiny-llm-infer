#include "block_manager.h"

#include <limits>

namespace my_vllm::engine
{
namespace
{
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;
}

BlockManager::BlockManager(int32_t num_blocks, int32_t block_size)
    : block_size_(block_size),
      blocks_(num_blocks > 0 ? static_cast<size_t>(num_blocks) : 0)
{
    for (int32_t id = 0; id < num_blocks; ++id)
    {
        blocks_[static_cast<size_t>(id)].id = id;
        free_ids_.insert(id);
    }
}

uint64_t BlockManager::compute_hash(const std::vector<int32_t>& token_ids,
                                    uint64_t prefix_hash)
{
    uint64_t hash = kFnvOffset;
    for (int shift = 0; shift < 64; shift += 8)
    {
        hash ^= static_cast<uint8_t>(prefix_hash >> shift);
        hash *= kFnvPrime;
    }
    for (const int32_t token : token_ids)
    {
        const uint32_t value = static_cast<uint32_t>(token);
        for (int shift = 0; shift < 32; shift += 8)
        {
            hash ^= static_cast<uint8_t>(value >> shift);
            hash *= kFnvPrime;
        }
    }
    return hash;
}

bool BlockManager::can_allocate(const Sequence& sequence, int32_t& cached_blocks) const
{
    cached_blocks = 0;
    const int32_t num_blocks = static_cast<int32_t>(sequence.num_blocks());
    int32_t new_blocks = num_blocks;
    uint64_t prefix_hash = 0;
    for (int32_t index = 0; index + 1 < num_blocks; ++index)
    {
        const std::vector<int32_t> tokens = sequence.block_tokens(static_cast<size_t>(index));
        if (tokens.size() != static_cast<size_t>(block_size_)) break;
        prefix_hash = compute_hash(tokens, prefix_hash);
        const auto found = hash_to_block_.find(prefix_hash);
        if (found == hash_to_block_.end()) break;
        const Block& block = blocks_.at(static_cast<size_t>(found->second));
        if (not block.hash_valid or block.token_ids != tokens) break;
        ++cached_blocks;
        if (block.ref_count > 0) --new_blocks;
    }
    return free_ids_.size() >= static_cast<size_t>(new_blocks);
}

int32_t BlockManager::allocate_block()
{
    if (free_ids_.empty()) return -1;
    const int32_t id = *free_ids_.begin();
    free_ids_.erase(free_ids_.begin());
    Block& block = blocks_.at(static_cast<size_t>(id));
    if (block.hash_valid)
    {
        const auto found = hash_to_block_.find(block.hash);
        if (found != hash_to_block_.end() and found->second == id)
        {
            hash_to_block_.erase(found);
        }
    }
    block.ref_count = 1;
    block.hash_valid = false;
    block.hash = 0;
    block.token_ids.clear();
    return id;
}

Status BlockManager::allocate(Sequence& sequence, int32_t cached_blocks)
{
    if (not sequence.block_table.empty())
    {
        return InvalidArgument("Cannot allocate blocks for an already allocated sequence.");
    }
    if (cached_blocks < 0 or cached_blocks > static_cast<int32_t>(sequence.num_blocks()))
    {
        return InvalidArgument("The cached block count is outside the sequence block range.");
    }
    uint64_t prefix_hash = 0;
    for (int32_t index = 0; index < cached_blocks; ++index)
    {
        const std::vector<int32_t> tokens = sequence.block_tokens(static_cast<size_t>(index));
        prefix_hash = compute_hash(tokens, prefix_hash);
        const auto found = hash_to_block_.find(prefix_hash);
        if (found == hash_to_block_.end())
        {
            return InternalError("A cached prefix block disappeared during allocation.");
        }
        Block& block = blocks_.at(static_cast<size_t>(found->second));
        if (not block.hash_valid or block.token_ids != tokens)
        {
            return InternalError("A prefix block hash collided with different token ids.");
        }
        if (block.ref_count == 0)
        {
            if (free_ids_.erase(block.id) != 1)
            {
                return InternalError("A cached free block is missing from the free list.");
            }
            block.ref_count = 1;
        }
        else
        {
            ++block.ref_count;
        }
        sequence.block_table.push_back(block.id);
    }
    for (size_t index = static_cast<size_t>(cached_blocks);
         index < sequence.num_blocks(); ++index)
    {
        const int32_t id = allocate_block();
        if (id < 0)
        {
            deallocate(sequence);
            return InternalError("The KV block pool ran out during allocation.");
        }
        sequence.block_table.push_back(id);
    }
    sequence.num_cached_tokens = static_cast<size_t>(cached_blocks) *
                                 static_cast<size_t>(block_size_);
    return Success();
}

void BlockManager::release_block(int32_t block_id)
{
    Block& block = blocks_.at(static_cast<size_t>(block_id));
    if (block.ref_count <= 0) return;
    --block.ref_count;
    if (block.ref_count == 0) free_ids_.insert(block_id);
}

void BlockManager::deallocate(Sequence& sequence)
{
    for (auto it = sequence.block_table.rbegin(); it != sequence.block_table.rend(); ++it)
    {
        release_block(*it);
    }
    sequence.block_table.clear();
    sequence.num_cached_tokens = 0;
    sequence.num_scheduled_tokens = 0;
}

bool BlockManager::can_append(const Sequence& sequence) const
{
    if (sequence.size() == 0) return false;
    const size_t position = sequence.size() - 1;
    const size_t block_index = position / static_cast<size_t>(block_size_);
    return block_index < sequence.block_table.size() or not free_ids_.empty();
}

Status BlockManager::may_append(Sequence& sequence)
{
    if (sequence.size() == 0)
    {
        return InvalidArgument("Cannot append a KV slot for an empty sequence.");
    }
    const size_t position = sequence.size() - 1;
    const size_t block_index = position / static_cast<size_t>(block_size_);
    if (block_index < sequence.block_table.size()) return Success();
    const int32_t id = allocate_block();
    if (id < 0) return InternalError("No KV block is available for the next token.");
    sequence.block_table.push_back(id);
    return Success();
}

Status BlockManager::hash_blocks(const Sequence& sequence)
{
    const size_t start = sequence.num_cached_tokens / static_cast<size_t>(block_size_);
    const size_t end = (sequence.num_cached_tokens + sequence.num_scheduled_tokens) /
                       static_cast<size_t>(block_size_);
    if (start >= end) return Success();
    if (end > sequence.block_table.size())
    {
        return InternalError("The sequence block table does not cover all completed KV tokens.");
    }

    uint64_t prefix_hash = 0;
    if (start > 0)
    {
        const Block& previous = blocks_.at(
            static_cast<size_t>(sequence.block_table.at(start - 1)));
        if (not previous.hash_valid)
        {
            return InternalError("Cannot hash a block whose prefix is not complete.");
        }
        prefix_hash = previous.hash;
    }

    for (size_t index = start; index < end; ++index)
    {
        const int32_t id = sequence.block_table.at(index);
        Block& block = blocks_.at(static_cast<size_t>(id));
        const std::vector<int32_t> tokens = sequence.block_tokens(index);
        if (tokens.size() != static_cast<size_t>(block_size_))
        {
            return InternalError("Only full KV blocks can be inserted into the prefix cache.");
        }
        const uint64_t hash = compute_hash(tokens, prefix_hash);
        block.hash = hash;
        block.hash_valid = true;
        block.token_ids = tokens;
        hash_to_block_[hash] = id;
        prefix_hash = hash;
    }
    return Success();
}

int32_t BlockManager::ref_count(int32_t block_id) const
{
    return blocks_.at(static_cast<size_t>(block_id)).ref_count;
}
}
