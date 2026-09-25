#ifndef MYVLLM_ENGINE_BLOCK_MANAGER_H_
#define MYVLLM_ENGINE_BLOCK_MANAGER_H_

#include <cstdint>
#include <set>
#include <unordered_map>
#include <vector>

#include "sequence.h"

namespace my_vllm::engine
{
class BlockManager
{
public:
    BlockManager(int32_t num_blocks, int32_t block_size);

    bool can_allocate(const Sequence& sequence, int32_t& cached_blocks) const;
    Status allocate(Sequence& sequence, int32_t cached_blocks);
    void deallocate(Sequence& sequence);
    bool can_append(const Sequence& sequence) const;
    Status may_append(Sequence& sequence);
    Status hash_blocks(const Sequence& sequence);

    int32_t block_size() const { return block_size_; }
    int32_t capacity() const { return static_cast<int32_t>(blocks_.size()); }
    int32_t free_count() const { return static_cast<int32_t>(free_ids_.size()); }
    int32_t ref_count(int32_t block_id) const;

private:
    struct Block
    {
        int32_t id = -1;
        int32_t ref_count = 0;
        bool hash_valid = false;
        uint64_t hash = 0;
        std::vector<int32_t> token_ids;
    };

    static uint64_t compute_hash(const std::vector<int32_t>& token_ids,
                                 uint64_t prefix_hash);
    int32_t allocate_block();
    void release_block(int32_t block_id);

    int32_t block_size_;
    std::vector<Block> blocks_;
    std::set<int32_t> free_ids_;
    std::unordered_map<uint64_t, int32_t> hash_to_block_;
};
}

#endif
