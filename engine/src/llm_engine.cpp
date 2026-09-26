#include "llm_engine.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace my_vllm::engine
{
LLMEngine::LLMEngine(ModelRunner& runner, EngineConfig config)
    : runner_(runner), config_(config), scheduler_(config.scheduler)
{
}

Status LLMEngine::init()
{
    if (config_.max_model_len <= 0 or config_.scheduler.max_num_seqs <= 0 or
        config_.scheduler.max_num_batched_tokens <= 0 or
        config_.scheduler.block_size <= 0 or config_.scheduler.num_kv_blocks <= 0)
    {
        return InvalidArgument("Engine capacities and block size must be positive.");
    }
    const Status status = runner_.configure_kv_cache(config_.scheduler.num_kv_blocks,
        config_.scheduler.block_size, config_.scheduler.max_num_batched_tokens);
    if (not status) return status;
    initialized_ = true;
    return Success();
}

Status LLMEngine::add_request(const std::string& prompt, SamplingParams sampling,
                              uint64_t& request_id)
{
    std::vector<int32_t> token_ids;
    const Status status = runner_.tokenize(prompt, token_ids);
    if (not status) return status;
    return add_request(std::move(token_ids), sampling, request_id);
}

Status LLMEngine::add_request(std::vector<int32_t> prompt_tokens, SamplingParams sampling,
                              uint64_t& request_id)
{
    if (not initialized_)
    {
        return InternalError("Initialize the engine before adding requests.");
    }
    const Status param_status = sampling.validate();
    if (not param_status) return param_status;
    if (sampling.max_tokens == 0)
    {
        return InvalidArgument("max_tokens must be greater than zero.");
    }
    if (prompt_tokens.empty())
    {
        return InvalidArgument("A prompt must encode to at least one token.");
    }
    if (prompt_tokens.size() > static_cast<size_t>(config_.max_model_len))
    {
        return InvalidArgument("The prompt exceeds max_model_len.");
    }
    const size_t max_completion = static_cast<size_t>(config_.max_model_len) -
                                  prompt_tokens.size() + 1;
    if (static_cast<size_t>(sampling.max_tokens) > max_completion)
    {
        return InvalidArgument("The requested completion exceeds max_model_len.");
    }

    request_id = next_request_id_++;
    auto sequence = std::make_shared<Sequence>(request_id, std::move(prompt_tokens),
                                                sampling, config_.scheduler.block_size);
    return scheduler_.add(sequence);
}

Status LLMEngine::sample(const std::vector<float>& logits, Sequence& sequence,
                         int32_t& token_id)
{
    if (logits.empty())
    {
        return InternalError("The model runner returned an empty logits vector.");
    }

    if (sequence.sampling.temperature <= 1.0e-6f)
    {
        token_id = static_cast<int32_t>(
            std::distance(logits.begin(), std::max_element(logits.begin(), logits.end())));
        return Success();
    }

    const float max_logit = *std::max_element(logits.begin(), logits.end());
    std::vector<double> weights(logits.size());
    const double inverse_temperature = 1.0 / sequence.sampling.temperature;
    for (size_t index = 0; index < logits.size(); ++index)
    {
        weights[index] = std::exp((static_cast<double>(logits[index]) - max_logit) *
                                  inverse_temperature);
    }

    if (sequence.sampling.top_p < 1.0f)
    {
        std::vector<size_t> order(logits.size());
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&weights](size_t left, size_t right) {
            return weights[left] > weights[right];
        });
        const double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        const double threshold = total * sequence.sampling.top_p;
        double cumulative = 0.0;
        bool reached_threshold = false;
        for (const size_t index : order)
        {
            if (reached_threshold)
            {
                weights[index] = 0.0;
            }
            else
            {
                cumulative += weights[index];
                reached_threshold = cumulative >= threshold;
            }
        }
    }

    std::discrete_distribution<size_t> distribution(weights.begin(), weights.end());
    token_id = static_cast<int32_t>(distribution(sequence.rng()));
    return Success();
}

Status LLMEngine::step(std::vector<GenerationResult>& completed)
{
    completed.clear();
    if (not initialized_)
    {
        return InternalError("Initialize the engine before stepping it.");
    }
    BatchPlan plan;
    Status status = scheduler_.schedule(plan);
    if (not status) return status;
    if (plan.sequences.empty()) return Success();

    std::vector<ModelBatchToken> batch_tokens;
    batch_tokens.reserve(plan.num_tokens);
    std::vector<size_t> sample_rows;
    std::vector<size_t> sample_sequence_indices;
    for (size_t index = 0; index < plan.sequences.size(); ++index)
    {
        const ScheduledSequence& item = plan.sequences[index];
        const Sequence& sequence = *item.sequence;
        const size_t end = item.token_start + item.token_count;
        for (size_t position = item.token_start; position < end; ++position)
        {
            batch_tokens.push_back({sequence.token_at(position),
                                    static_cast<int32_t>(position), sequence.block_table});
        }
        if (not plan.is_prefill or end == sequence.size())
        {
            sample_rows.push_back(batch_tokens.size() - 1);
            sample_sequence_indices.push_back(index);
        }
    }

    std::vector<std::vector<float>> sampled_logits;
    status = runner_.run_batch(batch_tokens, sample_rows, sampled_logits);
    if (not status) return status;
    if (sampled_logits.size() != sample_rows.size())
    {
        return InternalError("The model runner returned the wrong number of logits rows.");
    }
    std::vector<int32_t> sampled(plan.sequences.size(), -1);
    for (size_t index = 0; index < sampled_logits.size(); ++index)
    {
        const size_t sequence_index = sample_sequence_indices[index];
        status = sample(sampled_logits[index], *plan.sequences[sequence_index].sequence,
                        sampled[sequence_index]);
        if (not status) return status;
    }

    status = scheduler_.postprocess(plan, sampled, [this](int32_t token) {
        return runner_.is_eos(token);
    });
    if (not status) return status;
    for (const ScheduledSequence& item : plan.sequences)
    {
        if (item.sequence->status != SequenceStatus::kFinished) continue;
        GenerationResult result;
        result.request_id = item.sequence->id();
        result.token_ids = item.sequence->completion_tokens();
        result.text = runner_.decode(result.token_ids);
        completed.push_back(std::move(result));
    }
    return Success();
}

Status LLMEngine::generate(const std::vector<std::string>& prompts,
                           const std::vector<SamplingParams>& sampling,
                           std::vector<GenerationResult>& results)
{
    results.clear();
    if (prompts.empty()) return Success();
    if (not is_finished())
    {
        return InvalidArgument("Finish or step existing requests before calling generate.");
    }
    if (sampling.size() != 1 and sampling.size() != prompts.size())
    {
        return InvalidArgument("Pass one SamplingParams object or one per prompt.");
    }

    std::vector<std::vector<int32_t>> encoded_prompts;
    encoded_prompts.reserve(prompts.size());
    for (size_t index = 0; index < prompts.size(); ++index)
    {
        const SamplingParams& params = sampling.size() == 1 ? sampling.front() : sampling[index];
        const Status param_status = params.validate();
        if (not param_status) return param_status;
        if (params.max_tokens <= 0)
        {
            return InvalidArgument("max_tokens must be greater than zero.");
        }

        std::vector<int32_t> token_ids;
        const Status tokenize_status = runner_.tokenize(prompts[index], token_ids);
        if (not tokenize_status) return tokenize_status;
        if (token_ids.empty())
        {
            return InvalidArgument("A prompt must encode to at least one token.");
        }
        if (token_ids.size() > static_cast<size_t>(config_.max_model_len))
        {
            return InvalidArgument("A prompt exceeds max_model_len.");
        }
        const size_t max_completion = static_cast<size_t>(config_.max_model_len) -
                                      token_ids.size() + 1;
        if (static_cast<size_t>(params.max_tokens) > max_completion)
        {
            return InvalidArgument("A requested completion exceeds max_model_len.");
        }
        encoded_prompts.push_back(std::move(token_ids));
    }

    for (size_t index = 0; index < encoded_prompts.size(); ++index)
    {
        uint64_t request_id = 0;
        const SamplingParams& params = sampling.size() == 1 ? sampling.front() : sampling[index];
        const Status status = add_request(std::move(encoded_prompts[index]), params, request_id);
        if (not status) return status;
    }

    std::vector<GenerationResult> completed;
    while (not is_finished())
    {
        const Status status = step(completed);
        if (not status) return status;
        results.insert(results.end(), completed.begin(), completed.end());
    }
    std::sort(results.begin(), results.end(), [](const GenerationResult& left,
                                                 const GenerationResult& right) {
        return left.request_id < right.request_id;
    });
    return Success();
}

bool LLMEngine::is_finished() const
{
    return scheduler_.is_finished();
}
}
