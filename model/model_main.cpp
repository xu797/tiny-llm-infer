#include <algorithm>
#include <cmath>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>
#include <vector>

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
                 " [--gpu-memory-utilization <0..0.95>]\n";
}

int run_qwen3_engine(my_vllm::Qwen3Model& model,
                      my_vllm::DeviceType device_type,
                      const std::vector<std::string>& prompts, int max_new_tokens,
                      int max_seq_len, int requested_kv_blocks,
                      float gpu_memory_utilization)
{
    my_vllm::Status status = model.init(device_type);
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
                            num_kv_blocks, gpu_memory_utilization);
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

    testing::InitGoogleTest(&argc, argv);
    LOG(INFO) << "Start test...\n";
    return RUN_ALL_TESTS();
}
