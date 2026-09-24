#include <glog/logging.h>
#include <gtest/gtest.h>
#include <iostream>
#include <string>

#include "llama.h"
#include "qwen3.h"

namespace
{
void print_inference_usage(const char* program)
{
    std::cerr << "Usage: " << program
              << " --infer --tokenizer <tokenizer-file> --model <weights-file>"
                 " --prompt <text> [--model-type llama2|qwen3] [--device cpu|cuda]"
                 " [--max-new-tokens <count>] [--max-seq-len <count>] [--quant]\n";
}

template <typename ModelT>
int run_initialized_model(ModelT& model, my_vllm::DeviceType device_type,
                          const std::string& prompt, int max_new_tokens)
{
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

int run_inference(int argc, char* argv[])
{
    std::string tokenizer_path;
    std::string model_path;
    std::string prompt;
    std::string device = "cpu";
    std::string model_type = "llama2";
    int max_new_tokens = 64;
    int max_seq_len = 2048;
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
        else if (arg == "--model-type" && i + 1 < argc)
        {
            model_type = argv[++i];
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
        max_seq_len <= 0 || (device != "cpu" && device != "cuda") ||
        (model_type != "llama2" && model_type != "qwen3"))
    {
        print_inference_usage(argv[0]);
        return 2;
    }

    const my_vllm::DeviceType device_type = device == "cuda"
                                                ? my_vllm::DeviceType::kDeviceCUDA
                                                : my_vllm::DeviceType::kDeviceCPU;
    if (model_type == "qwen3")
    {
        if (quant)
        {
            std::cerr << "--quant applies only to the legacy Llama2 binary format.\n";
            return 2;
        }
        my_vllm::Qwen3Model model(tokenizer_path, model_path, max_seq_len);
        return run_initialized_model(model, device_type, prompt, max_new_tokens);
    }

    my_vllm::LLama2Model model(my_vllm::TokenizerType::kEncodeSpe, tokenizer_path, model_path, quant);
    return run_initialized_model(model, device_type, prompt, max_new_tokens);
}
} // namespace

TEST(Qwen3Tokenizer, ByteLevelUtf8RoundTrip)
{
    const std::string tokenizer_path = std::string(MYVLLM_SOURCE_DIR) +
                                       "/Qwen3-0.6B/tokenizer.json";
    my_vllm::Qwen3EncodeLayer tokenizer(tokenizer_path);
    const std::string text = "Hi there! 123\n你好，Qwen3🙂";
    const auto ids = tokenizer.encode(text);
    EXPECT_FALSE(ids.empty());
    EXPECT_EQ(tokenizer.decode(ids), text);
    EXPECT_EQ(tokenizer.encode("Hi"), std::vector<int32_t>{13048});
}

TEST(Qwen3Tokenizer, AddedChatTokensUseTheirConfiguredIds)
{
    const std::string tokenizer_path = std::string(MYVLLM_SOURCE_DIR) +
                                       "/Qwen3-0.6B/tokenizer.json";
    my_vllm::Qwen3EncodeLayer tokenizer(tokenizer_path);
    const auto ids = tokenizer.encode("<|im_start|>user<|im_end|>");
    ASSERT_GE(ids.size(), 3u);
    EXPECT_EQ(ids.front(), 151644);
    EXPECT_EQ(ids.back(), 151645);
    EXPECT_TRUE(tokenizer.is_sentence_ending(ids.back()));
}

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
