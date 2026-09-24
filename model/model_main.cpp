#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "llama.h"

namespace
{
void print_inference_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --infer --tokenizer <tokenizer.model> --model <weights.bin>"
                 " --prompt <text> [--device cpu|cuda] [--max-new-tokens <count>] [--quant]\n";
}

int run_inference(int argc, char* argv[])
{
    std::string tokenizer_path;
    std::string model_path;
    std::string prompt;
    std::string device = "cpu";
    int max_new_tokens = 64;
    bool quant = false;

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
            prompt = argv[++i];
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
        else if (arg == "--quant")
        {
            quant = true;
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

    if (tokenizer_path.empty() || model_path.empty() || prompt.empty() || max_new_tokens < 0 ||
        (device != "cpu" && device != "cuda"))
    {
        print_inference_usage(argv[0]);
        return 2;
    }

    my_vllm::LLama2Model model(my_vllm::TokenizerType::kEncodeSpe, tokenizer_path, model_path, quant);
    const my_vllm::DeviceType device_type = device == "cuda"
                                                ? my_vllm::DeviceType::kDeviceCUDA
                                                : my_vllm::DeviceType::kDeviceCPU;
    my_vllm::Status status = model.init(device_type);
    if (!status)
    {
        std::cerr << "Model initialization failed: " << status.get_err_msg() << '\n';
        return 1;
    }

    std::string generated;
    status = model.generate(prompt, max_new_tokens, generated);
    if (!status)
    {
        std::cerr << "Generation failed: " << status.get_err_msg() << '\n';
        return 1;
    }
    std::cout << generated << std::endl;
    return 0;
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
