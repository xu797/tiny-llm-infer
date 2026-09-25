#ifndef MYVLLM_ENGINE_LLM_ENGINE_H_
#define MYVLLM_ENGINE_LLM_ENGINE_H_

#include <cstdint>
#include <string>
#include <vector>

#include "model_runner.h"
#include "scheduler.h"

namespace my_vllm::engine
{
struct EngineConfig
{
    SchedulerConfig scheduler;
    int32_t max_model_len = 2048;
};

struct GenerationResult
{
    uint64_t request_id = 0;
    std::vector<int32_t> token_ids;
    std::string text;
};

class LLMEngine
{
public:
    LLMEngine(ModelRunner& runner, EngineConfig config);

    Status init();
    Status add_request(const std::string& prompt, SamplingParams sampling,
                       uint64_t& request_id);
    Status add_request(std::vector<int32_t> prompt_tokens, SamplingParams sampling,
                       uint64_t& request_id);
    Status step(std::vector<GenerationResult>& completed);
    Status generate(const std::vector<std::string>& prompts,
                    const std::vector<SamplingParams>& sampling,
                    std::vector<GenerationResult>& results);

    bool is_finished() const;
    const Scheduler& scheduler() const { return scheduler_; }

private:
    Status sample(const std::vector<float>& logits, Sequence& sequence, int32_t& token_id);

    ModelRunner& runner_;
    EngineConfig config_;
    Scheduler scheduler_;
    bool initialized_ = false;
    uint64_t next_request_id_ = 1;
};
}

#endif
