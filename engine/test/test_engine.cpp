#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>
#include <nlohmann/json.hpp>

#include "block_manager.h"
#include "cpu_alloc.h"
#include "llm_engine.h"
#include "matmul.h"
#include "paged_attention.h"
#include "qwen3.h"
#include "qwen3_runner.h"
#include "rmsnorm.h"

namespace
{
std::filesystem::path create_tiny_qwen3_checkpoint(int32_t vocab_size,
                                                            int32_t dominant_token)
{
    using json = nlohmann::json;
    struct TensorSpec
    {
        std::string name;
        std::vector<int32_t> shape;
        float value;
        size_t count;
    };

    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("tiny_qwen3_engine_test_" + std::to_string(::getpid()));
    std::filesystem::remove_all(directory);
    std::filesystem::create_directories(directory);

    const json config = {
        {"model_type", "qwen3"},
        {"hidden_size", 4},
        {"intermediate_size", 4},
        {"num_hidden_layers", 1},
        {"num_attention_heads", 2},
        {"num_key_value_heads", 1},
        {"head_dim", 2},
        {"vocab_size", vocab_size},
        {"max_position_embeddings", 16},
        {"rms_norm_eps", 1e-5},
        {"rope_theta", 10000.0},
        {"tie_word_embeddings", true}};
    std::ofstream config_file(directory / "config.json");
    config_file << config.dump();
    config_file.close();

    const std::vector<TensorSpec> specs = {
        {"model.embed_tokens.weight", {vocab_size, 4}, 0.f,
         static_cast<size_t>(vocab_size) * 4},
        {"model.layers.0.input_layernorm.weight", {4}, 1.f, 4},
        {"model.layers.0.self_attn.q_proj.weight", {4, 4}, 0.f, 16},
        {"model.layers.0.self_attn.k_proj.weight", {2, 4}, 0.f, 8},
        {"model.layers.0.self_attn.v_proj.weight", {2, 4}, 0.f, 8},
        {"model.layers.0.self_attn.o_proj.weight", {4, 4}, 0.f, 16},
        {"model.layers.0.self_attn.q_norm.weight", {2}, 1.f, 2},
        {"model.layers.0.self_attn.k_norm.weight", {2}, 1.f, 2},
        {"model.layers.0.post_attention_layernorm.weight", {4}, 1.f, 4},
        {"model.layers.0.mlp.gate_proj.weight", {4, 4}, 0.f, 16},
        {"model.layers.0.mlp.up_proj.weight", {4, 4}, 0.f, 16},
        {"model.layers.0.mlp.down_proj.weight", {4, 4}, 0.f, 16},
        {"model.norm.weight", {4}, 1.f, 4}};

    json header = json::object();
    size_t offset = 0;
    for (const TensorSpec& spec : specs)
    {
        const size_t bytes = spec.count * sizeof(float);
        header[spec.name] = {
            {"dtype", "F32"},
            {"shape", spec.shape},
            {"data_offsets", {offset, offset + bytes}}};
        offset += bytes;
    }
    std::string header_data = header.dump();
    header_data.append((8 - header_data.size() % 8) % 8, ' ');

    std::ofstream weights(directory / "model.safetensors", std::ios::binary);
    const uint64_t header_size = header_data.size();
    weights.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    weights.write(header_data.data(), static_cast<std::streamsize>(header_data.size()));
    for (const TensorSpec& spec : specs)
    {
        std::vector<float> values(spec.count, spec.value);
        if (spec.name == "model.embed_tokens.weight")
        {
            values[static_cast<size_t>(dominant_token) * 4] = 1.f;
        }
        else if (spec.value == 0.f)
        {
            for (size_t i = 0; i < values.size(); ++i)
            {
                const int32_t centered = static_cast<int32_t>((i * 13 + spec.name.size()) % 7) - 3;
                values[i] = static_cast<float>(centered) * 0.001f;
            }
        }
        weights.write(reinterpret_cast<const char*>(values.data()),
                      static_cast<std::streamsize>(values.size() * sizeof(float)));
    }
    weights.close();
    return directory;
}

class RecordingRunner final : public my_vllm::engine::ModelRunner
{
public:
    my_vllm::Status configure_kv_cache(int32_t blocks, int32_t block_size,
                                       int32_t max_batch_tokens) override
    {
        configured_blocks = blocks;
        configured_block_size = block_size;
        configured_batch_tokens = max_batch_tokens;
        return my_vllm::Success();
    }

    my_vllm::Status run_batch(
        const std::vector<my_vllm::engine::ModelBatchToken>& tokens,
        const std::vector<size_t>& sample_rows,
        std::vector<std::vector<float>>& logits) override
    {
        batches.push_back(tokens);
        sampled_rows.push_back(sample_rows);
        logits.clear();
        for (size_t row : sample_rows)
        {
            if (row >= tokens.size())
                return my_vllm::InvalidArgument("Sample row is outside the token batch.");
            std::vector<float> row_logits(4, 0.0f);
            row_logits[1] = 1.0f;
            logits.push_back(std::move(row_logits));
        }
        return my_vllm::Success();
    }

    my_vllm::Status tokenize(const std::string& prompt,
                             std::vector<int32_t>& token_ids) const override
    {
        token_ids.clear();
        for (unsigned char ch : prompt) token_ids.push_back(ch % 3);
        return my_vllm::Success();
    }

    std::string decode(const std::vector<int32_t>& token_ids) const override
    {
        return std::to_string(token_ids.size());
    }

    bool is_eos(int32_t token_id) const override { return token_id == 3; }

    int32_t configured_blocks = 0;
    int32_t configured_block_size = 0;
    int32_t configured_batch_tokens = 0;
    std::vector<std::vector<my_vllm::engine::ModelBatchToken>> batches;
    std::vector<std::vector<size_t>> sampled_rows;
};
}  // namespace

TEST(Qwen3Tokenizer, ByteLevelUtf8RoundTrip)
{
    const std::string tokenizer_path = std::string(MYVLLM_SOURCE_DIR) +
                                       "/Qwen3-0.6B/tokenizer.json";
    my_vllm::Qwen3EncodeLayer tokenizer(tokenizer_path);
    const std::string text = "Hi there! 123\n你好，Qwen3🙂";
    const auto ids = tokenizer.encode(text);
    EXPECT_FALSE(ids.empty());
    EXPECT_EQ(tokenizer.decode(ids), text);
    EXPECT_EQ(tokenizer.encode("Hi"), (std::vector<int32_t>{13048}));
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

TEST(EngineBatch, PrefillsMultiplePromptsInOneRunnerCall)
{
    using namespace my_vllm::engine;
    RecordingRunner runner;
    EngineConfig config;
    config.max_model_len = 16;
    config.scheduler.max_num_seqs = 4;
    config.scheduler.max_num_batched_tokens = 16;
    config.scheduler.block_size = 2;
    config.scheduler.num_kv_blocks = 16;

    LLMEngine engine(runner, config);
    ASSERT_TRUE(engine.init());
    EXPECT_EQ(runner.configured_batch_tokens, 16);

    SamplingParams sampling;
    sampling.max_tokens = 2;
    std::vector<GenerationResult> results;
    const my_vllm::Status invalid_batch =
        engine.generate({"abcd", ""}, {sampling}, results);
    EXPECT_FALSE(invalid_batch);
    EXPECT_TRUE(engine.is_finished());
    EXPECT_TRUE(runner.batches.empty());

    const my_vllm::Status status =
        engine.generate({"abcd", "xy"}, {sampling}, results);
    ASSERT_TRUE(status) << status.get_err_msg();

    ASSERT_EQ(runner.batches.size(), 2u);
    ASSERT_EQ(runner.batches[0].size(), 6u);
    EXPECT_EQ(runner.sampled_rows[0], (std::vector<size_t>{3, 5}));
    EXPECT_EQ(runner.batches[1].size(), 2u);
    EXPECT_EQ(runner.sampled_rows[1], (std::vector<size_t>{0, 1}));

    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].token_ids, (std::vector<int32_t>{1, 1}));
    EXPECT_EQ(results[1].token_ids, (std::vector<int32_t>{1, 1}));
}

TEST(EngineBatch, PreemptsAndReplaysWhenThePagePoolIsFull)
{
    using namespace my_vllm::engine;
    RecordingRunner runner;
    EngineConfig config;
    config.max_model_len = 4;
    config.scheduler.max_num_seqs = 2;
    config.scheduler.max_num_batched_tokens = 4;
    config.scheduler.block_size = 2;
    config.scheduler.num_kv_blocks = 2;

    LLMEngine engine(runner, config);
    ASSERT_TRUE(engine.init());
    SamplingParams sampling;
    sampling.max_tokens = 3;
    std::vector<GenerationResult> results;
    const my_vllm::Status status =
        engine.generate({"a", "b"}, {sampling}, results);
    ASSERT_TRUE(status) << status.get_err_msg();
    ASSERT_EQ(results.size(), 2u);
    EXPECT_EQ(results[0].token_ids.size(), 3u);
    EXPECT_EQ(results[1].token_ids.size(), 3u);
    EXPECT_GE(runner.batches.size(), 4u);
}

TEST(Qwen3Engine, RunsVariableLengthPromptsThroughPagedBatchPath)
{
    const std::string tokenizer_path =
        std::string(MYVLLM_SOURCE_DIR) + "/Qwen3-0.6B/tokenizer.json";
    my_vllm::Qwen3EncodeLayer tokenizer(tokenizer_path);
    std::string long_prompt = "Hi there!";
    std::vector<int32_t> long_prompt_tokens = tokenizer.encode(long_prompt);
    while (long_prompt_tokens.size() <= 8u)
    {
        long_prompt += " Hi";
        long_prompt_tokens = tokenizer.encode(long_prompt);
    }
    ASSERT_LE(long_prompt_tokens.size(), 16u);
    const int32_t expected_token = tokenizer.encode("Hi").front();
    const std::filesystem::path directory =
        create_tiny_qwen3_checkpoint(tokenizer.vocab_size(), expected_token);
    {
        my_vllm::Qwen3Model model(tokenizer_path,
                                  (directory / "model.safetensors").string(), 16);
        const my_vllm::Status init_status = model.init(my_vllm::DeviceType::kDeviceCPU);
        ASSERT_TRUE(init_status) << init_status.get_err_msg();
        EXPECT_EQ(model.paged_kv_cache_size_bytes(3, 2), 96u);

        my_vllm::engine::Qwen3ModelRunner runner(model);
        my_vllm::engine::EngineConfig config;
        config.max_model_len = 16;
        config.scheduler.max_num_seqs = 3;
        config.scheduler.max_num_batched_tokens = 8;
        config.scheduler.block_size = 2;
        config.scheduler.num_kv_blocks = 8;
        my_vllm::engine::LLMEngine engine(runner, config);
        const my_vllm::Status engine_status = engine.init();
        ASSERT_TRUE(engine_status) << engine_status.get_err_msg();

        my_vllm::engine::SamplingParams sampling;
        sampling.max_tokens = 1;
        std::vector<my_vllm::engine::GenerationResult> results;
        const my_vllm::Status generation_status =
            engine.generate({"Hi", "Hi there!", long_prompt}, {sampling}, results);
        ASSERT_TRUE(generation_status) << generation_status.get_err_msg();
        ASSERT_EQ(results.size(), 3u);
        EXPECT_EQ(results[0].token_ids, (std::vector<int32_t>{expected_token}));
        EXPECT_EQ(results[0].text, "Hi");
        EXPECT_FALSE(results[1].token_ids.empty());
        EXPECT_FALSE(results[2].token_ids.empty());
    }
    std::filesystem::remove_all(directory);
}

TEST(PagedBlockManager, SharesHashedFullPrefixPages)
{
    using namespace my_vllm::engine;
    BlockManager manager(4, 2);
    SamplingParams sampling;
    auto first = std::make_shared<Sequence>(1, std::vector<int32_t>{1, 2, 3, 4, 5},
                                            sampling, 2);
    int32_t cached_blocks = -1;
    ASSERT_TRUE(manager.can_allocate(*first, cached_blocks));
    EXPECT_EQ(cached_blocks, 0);
    ASSERT_TRUE(manager.allocate(*first, cached_blocks));
    const std::vector<int32_t> first_table = first->block_table;
    first->num_scheduled_tokens = 4;
    ASSERT_TRUE(manager.hash_blocks(*first));

    auto second = std::make_shared<Sequence>(2, std::vector<int32_t>{1, 2, 3, 4, 9},
                                             sampling, 2);
    ASSERT_TRUE(manager.can_allocate(*second, cached_blocks));
    EXPECT_EQ(cached_blocks, 2);
    ASSERT_TRUE(manager.allocate(*second, cached_blocks));
    EXPECT_EQ(second->num_cached_tokens, 4u);
    EXPECT_EQ(second->block_table[0], first_table[0]);
    EXPECT_EQ(second->block_table[1], first_table[1]);
    EXPECT_EQ(manager.ref_count(first_table[0]), 2);
    EXPECT_EQ(manager.ref_count(first_table[1]), 2);

    manager.deallocate(*first);
    manager.deallocate(*second);
    EXPECT_EQ(manager.free_count(), 4);
}

TEST(PagedAttention, FollowsLogicalPagesAndMasksFutureSlots)
{
    using namespace my_vllm;
    const auto allocator = CPUDeviceAllocatorFactory::get_instance();
    Tensor output(DataType::kDataTypeFp32, 2, true, allocator);
    Tensor query(DataType::kDataTypeFp32, 2, true, allocator);
    Tensor scores(DataType::kDataTypeFp32, 1, 4, true, allocator);
    Tensor keys(DataType::kDataTypeFp32, 1, 2, 2, 2, true, allocator);
    Tensor values(DataType::kDataTypeFp32, 1, 2, 2, 2, true, allocator);
    Tensor table(DataType::kDataTypeInt32, 2, true, allocator);

    query.ptr<float>()[0] = 0.f;
    query.ptr<float>()[1] = 0.f;
    std::fill(keys.ptr<float>(), keys.ptr<float>() + keys.size(), 0.f);
    std::fill(values.ptr<float>(), values.ptr<float>() + values.size(), 0.f);
    // Logical page 0 maps to physical page 1; logical page 1 maps to page 0.
    values.ptr<float>()[4] = 10.f;
    values.ptr<float>()[7] = 20.f;
    values.ptr<float>()[0] = 30.f;
    table.ptr<int32_t>()[0] = 1;
    table.ptr<int32_t>()[1] = 0;

    paged_attention_cpu(2, 1, 0, 2, 2, 4, 2, 1, 2, output, query, scores,
                        keys, values, table);
    EXPECT_NEAR(output.ptr<float>()[0], 40.f / 3.f, 1e-5f);
    EXPECT_NEAR(output.ptr<float>()[1], 20.f / 3.f, 1e-5f);
}

TEST(TensorSize, FourDimensionalKVPoolUsesWideProducts)
{
    using namespace my_vllm;
    // This is representative of [layers, pages, block_size, kv_dim] for
    // a 7B model's FP32 cache and exceeds INT32_MAX elements.
    Tensor cache(DataType::kDataTypeFp32, 28, 4982, 16, 1024);
    const size_t expected_elements =
        size_t{28} * size_t{4982} * size_t{16} * size_t{1024};
    EXPECT_EQ(cache.size(), expected_elements);
    EXPECT_EQ(cache.byte_size(), expected_elements * sizeof(float));
}

TEST(BatchedMatmul, MultipliesEveryTokenRow)
{
    using namespace my_vllm;
    const auto allocator = CPUDeviceAllocatorFactory::get_instance();
    Tensor input(DataType::kDataTypeFp32, 2, 3, true, allocator);
    Tensor weight(DataType::kDataTypeFp32, 2, 3, true, allocator);
    Tensor output(DataType::kDataTypeFp32, 2, 2, true, allocator);
    const float input_values[] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    const float weight_values[] = {1.f, 0.f, 0.f, 0.f, 1.f, 0.f};
    std::copy(input_values, input_values + 6, input.ptr<float>());
    std::copy(weight_values, weight_values + 6, weight.ptr<float>());

    MatmulLayer layer(DeviceType::kDeviceCPU, 2, 3);
    ASSERT_TRUE(layer.set_weight(0, weight));
    layer.set_input(0, input);
    layer.set_output(0, output);
    const Status status = layer.forward();
    ASSERT_TRUE(status) << status.get_err_msg();
    EXPECT_FLOAT_EQ(output.ptr<float>()[0], 1.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[1], 2.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[2], 4.f);
    EXPECT_FLOAT_EQ(output.ptr<float>()[3], 5.f);
}


TEST(test_rmsnorm_cpu, normalizes_each_row_independently)
{
    using namespace my_vllm;
    const auto allocator = CPUDeviceAllocatorFactory::get_instance();
    Tensor input(DataType::kDataTypeFp32, 2, 3, true, allocator);
    Tensor weight(DataType::kDataTypeFp32, 3, true, allocator);
    Tensor output(DataType::kDataTypeFp32, 2, 3, true, allocator);
    const float values[] = {3.f, 4.f, 0.f, 0.f, 0.f, 2.f};
    std::copy(values, values + 6, input.ptr<float>());
    std::fill(weight.ptr<float>(), weight.ptr<float>() + 3, 1.f);

    RmsNormLayer layer(DeviceType::kDeviceCPU, 3);
    ASSERT_TRUE(layer.set_weight(0, weight));
    layer.set_input(0, input);
    layer.set_output(0, output);
    const Status status = layer.forward();
    ASSERT_TRUE(status) << status.get_err_msg();

    const float first_scale = 1.f / std::sqrt((25.f / 3.f) + 1e-5f);
    const float second_scale = 1.f / std::sqrt((4.f / 3.f) + 1e-5f);
    EXPECT_NEAR(output.ptr<float>()[0], 3.f * first_scale, 1e-5f);
    EXPECT_NEAR(output.ptr<float>()[1], 4.f * first_scale, 1e-5f);
    EXPECT_NEAR(output.ptr<float>()[3], 0.f, 1e-5f);
    EXPECT_NEAR(output.ptr<float>()[5], 2.f * second_scale, 1e-5f);
}
