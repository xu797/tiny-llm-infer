#include "scheduler.h"

#include <algorithm>

namespace my_vllm::engine
{
Scheduler::Scheduler(SchedulerConfig config)
    : config_(config), block_manager_(config.num_kv_blocks, config.block_size)
{
}

Status Scheduler::add(const std::shared_ptr<Sequence>& sequence)
{
    if (not sequence or sequence->size() == 0)
    {
        return InvalidArgument("A scheduled request must contain at least one token.");
    }
    if (sequence->block_size() != config_.block_size)
    {
        return InvalidArgument("The sequence and block manager use different block sizes.");
    }
    waiting_.push_back(sequence);
    return Success();
}

bool Scheduler::is_finished() const
{
    return waiting_.empty() and running_.empty();
}

void Scheduler::preempt(const std::shared_ptr<Sequence>& sequence)
{
    block_manager_.deallocate(*sequence);
    sequence->status = SequenceStatus::kWaiting;
    sequence->is_prefill = true;
    sequence->num_scheduled_tokens = 0;
    waiting_.push_front(sequence);
}

Status Scheduler::schedule(BatchPlan& plan)
{
    plan = BatchPlan{};
    size_t batched_tokens = 0;

    while (not waiting_.empty() and
           plan.sequences.size() < static_cast<size_t>(config_.max_num_seqs))
    {
        const std::shared_ptr<Sequence> sequence = waiting_.front();
        if (sequence->block_table.empty())
        {
            int32_t cached_blocks = 0;
            if (not block_manager_.can_allocate(*sequence, cached_blocks))
            {
                if (sequence->num_blocks() > static_cast<size_t>(block_manager_.capacity()))
                {
                    return InvalidArgument("The prompt requires more KV blocks than the engine owns.");
                }
                break;
            }
            const Status status = block_manager_.allocate(*sequence, cached_blocks);
            if (not status) return status;
        }

        const size_t remaining_tokens = sequence->size() - sequence->num_cached_tokens;
        if (remaining_tokens == 0)
        {
            return InternalError("A waiting prompt has no uncached tokens to schedule.");
        }
        const size_t available = static_cast<size_t>(config_.max_num_batched_tokens) -
                                 batched_tokens;
        if (available == 0) break;
        if (available < remaining_tokens and not plan.sequences.empty()) break;

        const size_t scheduled = std::min(remaining_tokens, available);
        sequence->num_scheduled_tokens = scheduled;
        sequence->is_prefill = true;
        plan.sequences.push_back({sequence, sequence->num_cached_tokens, scheduled});
        batched_tokens += scheduled;

        if (sequence->num_cached_tokens + scheduled == sequence->size())
        {
            waiting_.pop_front();
            sequence->status = SequenceStatus::kRunning;
            running_.push_back(sequence);
        }
        else
        {
            break;
        }
    }

    if (not plan.sequences.empty())
    {
        plan.is_prefill = true;
        plan.num_tokens = batched_tokens;
        return Success();
    }

    bool preempted_sequence = false;
    const size_t candidates = running_.size();
    for (size_t index = 0; index < candidates and
         plan.sequences.size() < static_cast<size_t>(config_.max_num_seqs); ++index)
    {
        const std::shared_ptr<Sequence> sequence = running_.front();
        running_.pop_front();
        if (not block_manager_.can_append(*sequence))
        {
            preempt(sequence);
            preempted_sequence = true;
            continue;
        }

        const Status status = block_manager_.may_append(*sequence);
        if (not status)
        {
            preempt(sequence);
            preempted_sequence = true;
            continue;
        }
        sequence->num_scheduled_tokens = 1;
        sequence->is_prefill = false;
        plan.sequences.push_back({sequence, sequence->size() - 1, 1});
        running_.push_back(sequence);
    }

    plan.is_prefill = false;
    plan.num_tokens = plan.sequences.size();
    if (plan.sequences.empty() and not is_finished())
    {
        if (preempted_sequence) return schedule(plan);
        return InternalError("The scheduler could not make progress with the available KV blocks.");
    }
    return Success();
}

Status Scheduler::postprocess(const BatchPlan& plan,
                              const std::vector<int32_t>& sampled_tokens,
                              const std::function<bool(int32_t)>& is_eos)
{
    if (sampled_tokens.size() != plan.sequences.size())
    {
        return InvalidArgument("The model runner returned the wrong number of sampled tokens.");
    }

    for (size_t index = 0; index < plan.sequences.size(); ++index)
    {
        const std::shared_ptr<Sequence>& sequence = plan.sequences[index].sequence;
        Status status = block_manager_.hash_blocks(*sequence);
        if (not status) return status;
        sequence->num_cached_tokens += sequence->num_scheduled_tokens;
        sequence->num_scheduled_tokens = 0;

        if (plan.is_prefill and sequence->num_cached_tokens < sequence->size()) continue;
        const int32_t token = sampled_tokens[index];
        if (token < 0)
        {
            return InternalError("The model runner did not return a token for a completed step.");
        }
        sequence->append_token(token);

        const bool stopped_on_eos = not sequence->sampling.ignore_eos and is_eos(token);
        const bool reached_limit = sequence->completion_size() >=
                                   static_cast<size_t>(sequence->sampling.max_tokens);
        if (stopped_on_eos or reached_limit)
        {
            sequence->status = SequenceStatus::kFinished;
            block_manager_.deallocate(*sequence);
            running_.erase(std::remove(running_.begin(), running_.end(), sequence), running_.end());
        }
    }
    return Success();
}
}
