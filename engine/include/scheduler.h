#ifndef MYVLLM_ENGINE_SCHEDULER_H_
#define MYVLLM_ENGINE_SCHEDULER_H_

#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

#include "block_manager.h"

namespace my_vllm::engine
{
struct SchedulerConfig
{
    int32_t max_num_seqs = 16;
    int32_t max_num_batched_tokens = 1024;
    int32_t block_size = 16;
    int32_t num_kv_blocks = 128;
};

struct ScheduledSequence
{
    std::shared_ptr<Sequence> sequence;
    size_t token_start = 0;
    size_t token_count = 0;
};

struct BatchPlan
{
    bool is_prefill = false;
    size_t num_tokens = 0;
    std::vector<ScheduledSequence> sequences;
};

class Scheduler
{
public:
    explicit Scheduler(SchedulerConfig config);

    Status add(const std::shared_ptr<Sequence>& sequence);
    Status schedule(BatchPlan& plan);
    Status postprocess(const BatchPlan& plan, const std::vector<int32_t>& sampled_tokens,
                       const std::function<bool(int32_t)>& is_eos);
    bool is_finished() const;
    BlockManager& block_manager() { return block_manager_; }
    const BlockManager& block_manager() const { return block_manager_; }

private:
    void preempt(const std::shared_ptr<Sequence>& sequence);

    SchedulerConfig config_;
    BlockManager block_manager_;
    std::deque<std::shared_ptr<Sequence>> waiting_;
    std::deque<std::shared_ptr<Sequence>> running_;
};
}

#endif
