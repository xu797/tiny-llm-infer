#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#include "qwen3.h"
#include "llm_engine.h"
#include "qwen3_runner.h"

namespace
{
void print_inference_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --infer --tokenizer <tokenizer-file> --model <weights-file>"
                 " --prompt <text> [--prompt <text> ...] [--device cpu|cuda]"
                 " [--max-new-tokens <count>] [--max-seq-len <count>] [--num-kv-blocks <count>]"
                 " [--gpu-memory-utilization <0..0.95>] [--quantization none|int8]\n";
}

void print_benchmark_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --bench --tokenizer <tokenizer-file> --model <weights-file>"
                 " [--device cpu|cuda] [--num-seqs N] [--min-input-len N]"
                 " [--max-input-len N] [--min-output-len N] [--max-output-len N]"
                 " [--warmup N] [--repeats N] [--max-seq-len N]"
                 " [--num-kv-blocks N] [--gpu-memory-utilization F]"
                 " [--max-token-id N] [--seed N] [--quantization none|int8]"
                 " [--manifest-prefix PATH]\n";
}

struct BenchmarkOptions
{
    std::string tokenizer_path;
    std::string model_path;
    std::string device = "cuda";
    std::string quantization = "none";
    std::string manifest_prefix;
    int32_t num_seqs = 8;
    int32_t min_input_len = 64;
    int32_t max_input_len = 128;
    int32_t min_output_len = 32;
    int32_t max_output_len = 64;
    int32_t warmup = 1;
    int32_t repeats = 3;
    int32_t max_seq_len = 256;
    int32_t num_kv_blocks = 0;
    int32_t max_token_id = 10000;
    uint64_t seed = 0;
    float gpu_memory_utilization = 0.7f;
};

bool parse_integer(const char* text, int64_t& value)
{
    try
    {
        size_t consumed = 0;
        value = std::stoll(text, &consumed);
        return consumed == std::string(text).size();
    }
    catch (const std::exception&)
    {
        return false;
    }
}

int parse_benchmark_options(int argc, char* argv[], BenchmarkOptions& options)
{
    for (int index = 2; index < argc; ++index)
    {
        const std::string arg = argv[index];
        if (arg == "--help" || arg == "-h")
        {
            print_benchmark_usage(argv[0]);
            return 0;
        }
        if (index + 1 >= argc)
        {
            print_benchmark_usage(argv[0]);
            return 2;
        }
        const std::string value = argv[++index];
        if (arg == "--tokenizer") options.tokenizer_path = value;
        else if (arg == "--model") options.model_path = value;
        else if (arg == "--device") options.device = value;
        else if (arg == "--quantization") options.quantization = value;
        else if (arg == "--manifest-prefix") options.manifest_prefix = value;
        else if (arg == "--gpu-memory-utilization")
        {
            try { options.gpu_memory_utilization = std::stof(value); }
            catch (const std::exception&) { return 2; }
        }
        else if (arg == "--seed")
        {
            int64_t parsed = 0;
            if (!parse_integer(value.c_str(), parsed) || parsed < 0) return 2;
            options.seed = static_cast<uint64_t>(parsed);
        }
        else
        {
            int64_t parsed = 0;
            if (!parse_integer(value.c_str(), parsed) || parsed < 0 || parsed > INT32_MAX)
                return 2;
            if (arg == "--num-seqs") options.num_seqs = static_cast<int32_t>(parsed);
            else if (arg == "--min-input-len") options.min_input_len = static_cast<int32_t>(parsed);
            else if (arg == "--max-input-len") options.max_input_len = static_cast<int32_t>(parsed);
            else if (arg == "--min-output-len") options.min_output_len = static_cast<int32_t>(parsed);
            else if (arg == "--max-output-len") options.max_output_len = static_cast<int32_t>(parsed);
            else if (arg == "--warmup") options.warmup = static_cast<int32_t>(parsed);
            else if (arg == "--repeats") options.repeats = static_cast<int32_t>(parsed);
            else if (arg == "--max-seq-len") options.max_seq_len = static_cast<int32_t>(parsed);
            else if (arg == "--num-kv-blocks") options.num_kv_blocks = static_cast<int32_t>(parsed);
            else if (arg == "--max-token-id") options.max_token_id = static_cast<int32_t>(parsed);
            else
            {
                print_benchmark_usage(argv[0]);
                return 2;
            }
        }
    }

    if (options.tokenizer_path.empty() || options.model_path.empty() ||
        (options.device != "cpu" && options.device != "cuda") ||
        options.num_seqs <= 0 || options.min_input_len <= 0 ||
        options.max_input_len < options.min_input_len || options.min_output_len <= 0 ||
        options.max_output_len < options.min_output_len || options.warmup < 0 ||
        options.repeats <= 0 || options.max_seq_len <= 0 || options.num_kv_blocks < 0 ||
        options.max_token_id < 0 ||
        (options.quantization != "none" && options.quantization != "int8") ||
        !std::isfinite(options.gpu_memory_utilization) ||
        options.gpu_memory_utilization <= 0.0f || options.gpu_memory_utilization > 0.95f ||
        static_cast<int64_t>(options.max_input_len) + options.max_output_len >
            static_cast<int64_t>(options.max_seq_len) + 1)
    {
        print_benchmark_usage(argv[0]);
        return 2;
    }
    return -1;
}

std::vector<std::pair<std::vector<int32_t>, my_vllm::engine::SamplingParams>>
make_benchmark_requests(const BenchmarkOptions& options, uint64_t seed_value,
                       int32_t vocab_size)
{
    std::mt19937_64 rng(seed_value);
    std::uniform_int_distribution<int32_t> input_length(options.min_input_len,
                                                        options.max_input_len);
    std::uniform_int_distribution<int32_t> output_length(options.min_output_len,
                                                         options.max_output_len);
    std::uniform_int_distribution<int32_t> token_id(
        0, std::min(options.max_token_id, vocab_size - 1));
    std::vector<std::pair<std::vector<int32_t>, my_vllm::engine::SamplingParams>> requests;
    requests.reserve(static_cast<size_t>(options.num_seqs));
    for (int32_t index = 0; index < options.num_seqs; ++index)
    {
        std::vector<int32_t> prompt(static_cast<size_t>(input_length(rng)));
        std::generate(prompt.begin(), prompt.end(), [&] { return token_id(rng); });
        my_vllm::engine::SamplingParams sampling;
        sampling.temperature = 0.6f;
        sampling.ignore_eos = true;
        sampling.max_tokens = output_length(rng);
        sampling.seed = seed_value + static_cast<uint64_t>(index) + 1;
        requests.emplace_back(std::move(prompt), sampling);
    }
    return requests;
}

bool write_benchmark_manifest(
    const std::string& prefix, int32_t run_index,
    const std::vector<std::pair<std::vector<int32_t>, my_vllm::engine::SamplingParams>>& requests)
{
    if (prefix.empty()) return true;
    nlohmann::json manifest;
    manifest["requests"] = nlohmann::json::array();
    for (const auto& request : requests)
    {
        nlohmann::json item;
        item["prompt_token_ids"] = request.first;
        item["max_tokens"] = request.second.max_tokens;
        item["temperature"] = request.second.temperature;
        item["top_p"] = request.second.top_p;
        item["ignore_eos"] = request.second.ignore_eos;
        item["seed"] = request.second.seed;
        manifest["requests"].push_back(std::move(item));
    }
    const std::string path = prefix + "_run" + std::to_string(run_index) + ".json";
    std::ofstream output(path);
    if (!output) return false;
    output << manifest.dump(2) << '\n';
    return output.good();
}

my_vllm::Status run_benchmark_requests(
    my_vllm::engine::LLMEngine& engine,
    const std::vector<std::pair<std::vector<int32_t>, my_vllm::engine::SamplingParams>>& requests,
    uint64_t& actual_output_tokens, const std::string& phase)
{
    actual_output_tokens = 0;
    const uint64_t generated_tokens_before = engine.total_generated_tokens();
    std::cerr << phase << ": submitting " << requests.size() << " requests...\n";
    for (const auto& request : requests)
    {
        uint64_t request_id = 0;
        const my_vllm::Status status = engine.add_request(request.first, request.second,
                                                          request_id);
        if (!status) return status;
    }
    std::cerr << phase << ": running model steps...\n";
    std::vector<my_vllm::engine::GenerationResult> completed;
    size_t completed_requests = 0;
    size_t step_count = 0;
    while (!engine.is_finished())
    {
        const my_vllm::Status status = engine.step(completed);
        if (!status) return status;
        ++step_count;
        completed_requests += completed.size();
        for (const auto& result : completed)
            actual_output_tokens += result.token_ids.size();
        if (step_count == 1 || step_count % 10 == 0)
        {
            std::cerr << phase << ": steps=" << step_count << " completed_requests="
                      << completed_requests << "/" << requests.size()
                      << " generated_tokens="
                      << engine.total_generated_tokens() - generated_tokens_before << '\n';
        }
    }
    if (completed_requests != requests.size())
        return my_vllm::InternalError("The benchmark engine returned an incomplete result set.");
    if (engine.total_generated_tokens() - generated_tokens_before != actual_output_tokens)
        return my_vllm::InternalError("The benchmark token counters do not match.");
    return my_vllm::Success();
}

int run_benchmark(const BenchmarkOptions& options)
{
    const my_vllm::DeviceType device_type = options.device == "cuda"
        ? my_vllm::DeviceType::kDeviceCUDA : my_vllm::DeviceType::kDeviceCPU;
    my_vllm::Qwen3Model model(options.tokenizer_path, options.model_path,
                             options.max_seq_len);
    my_vllm::Status status;
    status = model.set_weight_quantization(options.quantization == "int8" ? 128 : 0);
    if (!status)
    {
        std::cerr << "Quantization setup failed: " << status.get_err_msg() << '\n';
        return 2;
    }
    status = model.init(device_type);
    if (!status)
    {
        std::cerr << "Model initialization failed: " << status.get_err_msg() << '\n';
        return 1;
    }
    if (model.vocab_size() <= 0)
    {
        std::cerr << "Model vocabulary size is invalid.\n";
        return 1;
    }
    if (options.max_token_id >= model.vocab_size())
    {
        std::cerr << "--max-token-id must be smaller than the model vocabulary size.\n";
        return 2;
    }

    my_vllm::engine::Qwen3ModelRunner runner(model);
    my_vllm::engine::EngineConfig config;
    config.max_model_len = options.max_seq_len;
    config.scheduler.max_num_seqs = options.num_seqs;
    config.scheduler.max_num_batched_tokens = std::min<int32_t>(options.max_seq_len, 256);
    config.scheduler.block_size = std::min<int32_t>(16, options.max_seq_len);
    if (options.num_kv_blocks > 0)
    {
        config.scheduler.num_kv_blocks = options.num_kv_blocks;
    }
    else if (device_type == my_vllm::DeviceType::kDeviceCUDA)
    {
        status = model.estimate_paged_kv_cache_blocks(
            config.scheduler.block_size, config.scheduler.max_num_batched_tokens,
            options.gpu_memory_utilization, config.scheduler.num_kv_blocks);
        if (!status)
        {
            std::cerr << "KV cache sizing failed: " << status.get_err_msg() << '\n';
            return 1;
        }
        const int64_t max_tokens_per_sequence =
            static_cast<int64_t>(options.max_input_len) + options.max_output_len;
        const int64_t blocks_per_sequence =
            (max_tokens_per_sequence + config.scheduler.block_size - 1) /
            config.scheduler.block_size;
        const int64_t workload_blocks = blocks_per_sequence * options.num_seqs;
        config.scheduler.num_kv_blocks = std::min<int32_t>(
            config.scheduler.num_kv_blocks,
            static_cast<int32_t>(std::min<int64_t>(workload_blocks, INT32_MAX)));
    }
    else
    {
        config.scheduler.num_kv_blocks =
            (options.max_seq_len + config.scheduler.block_size - 1) /
            config.scheduler.block_size;
    }

    const size_t kv_cache_bytes = model.paged_kv_cache_size_bytes(
        config.scheduler.num_kv_blocks, config.scheduler.block_size);
    std::cerr << "Preallocating " << config.scheduler.num_kv_blocks
              << " KV blocks (" << (kv_cache_bytes >> 20) << " MiB).\n";
    my_vllm::engine::LLMEngine engine(runner, config);
    std::cerr << "Initializing engine and allocating the KV cache...\n";
    status = engine.init();
    if (!status)
    {
        std::cerr << "Engine initialization failed: " << status.get_err_msg() << '\n';
        return 1;
    }
    std::cerr << "Engine initialized.\n";

    for (int32_t index = 0; index < options.warmup; ++index)
    {
        const auto requests = make_benchmark_requests(
            options, options.seed + static_cast<uint64_t>(index) + 1, model.vocab_size());
        uint64_t warmup_tokens = 0;
        status = run_benchmark_requests(engine, requests, warmup_tokens,
                                       "warmup " + std::to_string(index + 1));
        if (!status)
        {
            std::cerr << "Warmup failed: " << status.get_err_msg() << '\n';
            return 1;
        }
        std::cerr << "Warmup " << (index + 1) << " complete.\n";
    }

    double total_elapsed = 0.0;
    uint64_t total_output_tokens = 0;
    for (int32_t index = 0; index < options.repeats; ++index)
    {
        const auto requests = make_benchmark_requests(
            options, options.seed + 1000 + static_cast<uint64_t>(index),
            model.vocab_size());
        if (!write_benchmark_manifest(options.manifest_prefix, index + 1, requests))
        {
            std::cerr << "Failed to write benchmark request manifest for run "
                      << (index + 1) << ".\n";
            return 1;
        }
        uint64_t actual_tokens = 0;
        const auto started = std::chrono::steady_clock::now();
        status = run_benchmark_requests(engine, requests, actual_tokens,
                                       "run " + std::to_string(index + 1));
        const auto stopped = std::chrono::steady_clock::now();
        if (!status)
        {
            std::cerr << "Benchmark run failed: " << status.get_err_msg() << '\n';
            return 1;
        }
        const double elapsed = std::chrono::duration<double>(stopped - started).count();
        total_elapsed += elapsed;
        total_output_tokens += actual_tokens;
        std::cout << "run=" << (index + 1) << " requests=" << options.num_seqs
                  << " output_tokens=" << actual_tokens << " elapsed_seconds=" << elapsed
                  << " output_tokens_per_second=" << actual_tokens / elapsed << '\n';
    }
    std::cout << "aggregate_runs=" << options.repeats
              << " output_tokens=" << total_output_tokens
              << " elapsed_seconds=" << total_elapsed
              << " output_tokens_per_second=" << total_output_tokens / total_elapsed
              << '\n';
    return 0;
}

int run_qwen3_engine(my_vllm::Qwen3Model& model,
                      my_vllm::DeviceType device_type,
                      const std::vector<std::string>& prompts, int max_new_tokens,
                      int max_seq_len, int requested_kv_blocks,
                      float gpu_memory_utilization, int32_t quant_group_size)
{
    my_vllm::Status status = model.set_weight_quantization(quant_group_size);
    if (!status)
    {
        std::cerr << "Quantization setup failed: " << status.get_err_msg() << '\n';
        return 2;
    }
    status = model.init(device_type);
    if (!status)
    {
        std::cerr << "Model initialization failed: " << status.get_err_msg() << '\n';
        return 1;
    }
    if (max_new_tokens == 0)
    {
        for (size_t i = 0; i < prompts.size(); ++i) std::cout << std::endl;
        return 0;
    }

    my_vllm::engine::Qwen3ModelRunner runner(model);
    my_vllm::engine::EngineConfig config;
    config.max_model_len = max_seq_len;
    config.scheduler.max_num_seqs =
        std::min<int32_t>(16, static_cast<int32_t>(prompts.size()));
    config.scheduler.max_num_batched_tokens = std::min(max_seq_len, 256);
    config.scheduler.block_size = std::min(16, max_seq_len);
    if (requested_kv_blocks > 0)
    {
        config.scheduler.num_kv_blocks = requested_kv_blocks;
    }
    else if (device_type == my_vllm::DeviceType::kDeviceCUDA)
    {
        status = model.estimate_paged_kv_cache_blocks(
            config.scheduler.block_size, config.scheduler.max_num_batched_tokens,
            gpu_memory_utilization, config.scheduler.num_kv_blocks);
        if (!status)
        {
            std::cerr << "KV cache sizing failed: " << status.get_err_msg() << '\n';
            return 1;
        }
    }
    else
    {
        config.scheduler.num_kv_blocks =
            (max_seq_len + config.scheduler.block_size - 1) / config.scheduler.block_size;
    }
    const size_t kv_cache_bytes = model.paged_kv_cache_size_bytes(
        config.scheduler.num_kv_blocks, config.scheduler.block_size);
    if (kv_cache_bytes == 0)
    {
        std::cerr << "The requested KV cache size is invalid.\n";
        return 2;
    }
    std::cerr << "Preallocating " << config.scheduler.num_kv_blocks
              << " KV blocks (" << (kv_cache_bytes >> 20) << " MiB).\n";
    my_vllm::engine::LLMEngine engine(runner, config);
    status = engine.init();
    if (!status)
    {
        std::cerr << "Engine initialization failed: " << status.get_err_msg() << '\n';
        return 1;
    }

    my_vllm::engine::SamplingParams sampling;
    sampling.max_tokens = max_new_tokens;
    std::vector<my_vllm::engine::GenerationResult> results;
    status = engine.generate(prompts, {sampling}, results);
    if (!status)
    {
        std::cerr << "Generation failed: " << status.get_err_msg() << '\n';
        return 1;
    }
    if (results.size() != prompts.size())
    {
        std::cerr << "The engine returned an incomplete result set.\n";
        return 1;
    }
    for (const auto& result : results) std::cout << result.text << std::endl;
    return 0;
}


int run_inference(int argc, char* argv[])
{
    std::string tokenizer_path;
    std::string model_path;
    std::vector<std::string> prompts;
    std::string device = "cpu";
    int max_new_tokens = 64;
    int max_seq_len = 2048;
    int num_kv_blocks = 0;
    float gpu_memory_utilization = 0.9f;
    std::string quantization = "none";

    for (int i = 2; i < argc; ++i)
    {
        const std::string arg = argv[i];
        if (arg == "--tokenizer" && i + 1 < argc)
        {
            tokenizer_path = argv[++i];
        }
        else if (arg == "--model" && i + 1 < argc)
        {
            model_path = argv[++i];
        }
        else if (arg == "--prompt" && i + 1 < argc)
        {
            prompts.emplace_back(argv[++i]);
        }
        else if (arg == "--device" && i + 1 < argc)
        {
            device = argv[++i];
        }
        else if (arg == "--quantization" && i + 1 < argc)
        {
            quantization = argv[++i];
        }
        else if (arg == "--max-new-tokens" && i + 1 < argc)
        {
            try
            {
                max_new_tokens = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--max-new-tokens must be an integer.\n";
                return 2;
            }
        }
        else if (arg == "--max-seq-len" && i + 1 < argc)
        {
            try
            {
                max_seq_len = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--max-seq-len must be an integer.\n";
                return 2;
            }
        }
        else if (arg == "--num-kv-blocks" && i + 1 < argc)
        {
            try
            {
                num_kv_blocks = std::stoi(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--num-kv-blocks must be an integer.\n";
                return 2;
            }
        }
        else if (arg == "--gpu-memory-utilization" && i + 1 < argc)
        {
            try
            {
                gpu_memory_utilization = std::stof(argv[++i]);
            }
            catch (const std::exception&)
            {
                std::cerr << "--gpu-memory-utilization must be a number.\n";
                return 2;
            }
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_inference_usage(argv[0]);
            return 0;
        }
        else
        {
            std::cerr << "Unknown or incomplete inference option: " << arg << '\n';
            print_inference_usage(argv[0]);
            return 2;
        }
    }

    if (tokenizer_path.empty() || model_path.empty() || prompts.empty() ||
        std::any_of(prompts.begin(), prompts.end(), [](const std::string& value) {
            return value.empty();
        }) || max_new_tokens < 0 || max_seq_len <= 0 || num_kv_blocks < 0 ||
        !std::isfinite(gpu_memory_utilization) || gpu_memory_utilization <= 0.0f ||
        gpu_memory_utilization > 0.95f ||
        (quantization != "none" && quantization != "int8") ||
        (device != "cpu" && device != "cuda"))
    {
        print_inference_usage(argv[0]);
        return 2;
    }

    const my_vllm::DeviceType device_type = device == "cuda"
                                                ? my_vllm::DeviceType::kDeviceCUDA
                                                : my_vllm::DeviceType::kDeviceCPU;
    my_vllm::Qwen3Model model(tokenizer_path, model_path, max_seq_len);
    return run_qwen3_engine(model, device_type, prompts, max_new_tokens, max_seq_len,
                            num_kv_blocks, gpu_memory_utilization,
                            quantization == "int8" ? 128 : 0);
}
} // namespace

int main(int argc, char *argv[]) 
{
    google::InitGoogleLogging("my_vllm");
    FLAGS_alsologtostderr = true;

    if (argc > 1 && std::string(argv[1]) == "--infer")
    {
        return run_inference(argc, argv);
    }
    if (argc > 1 && std::string(argv[1]) == "--bench")
    {
        BenchmarkOptions options;
        const int parsed = parse_benchmark_options(argc, argv, options);
        if (parsed >= 0) return parsed;
        return run_benchmark(options);
    }

    testing::InitGoogleTest(&argc, argv);
    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}
