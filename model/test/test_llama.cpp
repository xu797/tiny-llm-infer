#include <filesystem>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <unistd.h>

#include <gtest/gtest.h>
#include <sentencepiece_processor.h>

#include "config.h"
#include "cpu_alloc.h"
#include "llama.h"
#include "rmsnorm.h"

namespace
{
void write_floats(std::ofstream& file, size_t count, float value)
{
    const std::vector<float> values(count, value);
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
}

std::string create_tiny_llama_checkpoint(int32_t vocab_size)
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("tiny_llama_test_" + std::to_string(::getpid()) + ".bin");
    std::ofstream file(path, std::ios::binary);
    if (!file)
    {
        return {};
    }

    constexpr int32_t dim = 4;
    constexpr int32_t hidden_dim = 8;
    constexpr int32_t layer_num = 1;
    constexpr int32_t head_num = 2;
    constexpr int32_t kv_head_num = 2;
    constexpr int32_t seq_len = 16;
    const my_vllm::ModelConfig config{dim, hidden_dim, layer_num, head_num,
                                     kv_head_num, vocab_size, seq_len};
    file.write(reinterpret_cast<const char*>(&config), sizeof(config));

    // Legacy Llama2 weight order: embeddings, attention norm, attention matrices,
    // FFN norm, FFN matrices, final norm, RoPE tables. Positive vocab ties lm_head.
    write_floats(file, static_cast<size_t>(vocab_size) * dim, 0.f);
    write_floats(file, dim, 1.f);                     // attention_norm
    write_floats(file, dim * dim, 0.f);               // wq
    write_floats(file, dim * dim, 0.f);               // wk
    write_floats(file, dim * dim, 0.f);               // wv
    write_floats(file, dim * dim, 0.f);               // wo
    write_floats(file, dim, 1.f);                     // ffn_norm
    write_floats(file, hidden_dim * dim, 0.f);        // w1
    write_floats(file, dim * hidden_dim, 0.f);        // w2
    write_floats(file, hidden_dim * dim, 0.f);        // w3
    write_floats(file, dim, 1.f);                     // final norm
    write_floats(file, seq_len * (dim / head_num), 0.f); // freqs_cos
    write_floats(file, seq_len * (dim / head_num), 0.f); // freqs_sin
    file.close();
    return path.string();
}
} // namespace

TEST(test_rmsnorm_cpu, normalizes_each_row_independently)
{
    using namespace my_vllm;
    const auto allocator = CPUDeviceAllocatorFactory::get_instance();
    Tensor input(DataType::kDataTypeFp32, 2, 3, true, allocator);
    Tensor weight(DataType::kDataTypeFp32, 3, true, allocator);
    Tensor output(DataType::kDataTypeFp32, 2, 3, true, allocator);
    const float values[] = {3.f, 4.f, 0.f, 0.f, 0.f, 2.f};
    for (int i = 0; i < 6; ++i)
    {
        input.index<float>(i) = values[i];
    }
    for (int i = 0; i < 3; ++i)
    {
        weight.index<float>(i) = 1.f;
    }

    RmsNormLayer layer(DeviceType::kDeviceCPU, 3);
    ASSERT_TRUE(layer.set_weight(0, weight));
    layer.set_input(0, input);
    layer.set_output(0, output);
    const Status status = layer.forward();
    ASSERT_TRUE(status) << status.get_err_msg();

    const float first_scale = 1.f / std::sqrt((25.f / 3.f) + 1e-5f);
    const float second_scale = 1.f / std::sqrt((4.f / 3.f) + 1e-5f);
    EXPECT_NEAR(output.index<float>(0), 3.f * first_scale, 1e-5f);
    EXPECT_NEAR(output.index<float>(1), 4.f * first_scale, 1e-5f);
    EXPECT_NEAR(output.index<float>(3), 0.f, 1e-5f);
    EXPECT_NEAR(output.index<float>(5), 2.f * second_scale, 1e-5f);
}

TEST(test_llama_model, cpu_generate_with_tiny_legacy_checkpoint)
{
    const std::string tokenizer_path =
        std::string(MYVLLM_SOURCE_DIR) + "/llama2/llama2/tokenizer.model";
    sentencepiece::SentencePieceProcessor tokenizer;
    const auto tokenizer_status = tokenizer.Load(tokenizer_path);
    ASSERT_TRUE(tokenizer_status.ok()) << tokenizer_status.ToString();

    const std::string checkpoint_path = create_tiny_llama_checkpoint(tokenizer.GetPieceSize());
    ASSERT_FALSE(checkpoint_path.empty());
    {
        my_vllm::LLama2Model model(my_vllm::TokenizerType::kEncodeSpe,
                                   tokenizer_path, checkpoint_path, false);
        const my_vllm::Status init_status = model.init(my_vllm::DeviceType::kDeviceCPU);
        ASSERT_TRUE(init_status) << init_status.get_err_msg();

        std::string generated;
        const my_vllm::Status generation_status = model.generate("Hi", 3, generated);
        ASSERT_TRUE(generation_status) << generation_status.get_err_msg();
        EXPECT_EQ(generated, model.decode(std::vector<int32_t>(3, 0)));

        const my_vllm::Status context_status = model.generate("Hi", 100, generated);
        EXPECT_FALSE(context_status);
        EXPECT_TRUE(generated.empty());
    }
    std::filesystem::remove(checkpoint_path);
}
