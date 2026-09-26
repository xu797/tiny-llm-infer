#ifndef MYVLLM_MODEL_QWEN3_H_
#define MYVLLM_MODEL_QWEN3_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda_config.h"
#include "embedding.h"
#include "model.h"

namespace my_vllm
{

// Qwen3 keeps the weights and operations for each decoder block together.
struct Qwen3DecoderLayer
{
    std::shared_ptr<Layer> input_norm_;
    std::shared_ptr<Layer> q_proj_;
    std::shared_ptr<Layer> k_proj_;
    std::shared_ptr<Layer> v_proj_;
    std::shared_ptr<Layer> o_proj_;
    std::shared_ptr<Layer> q_norm_;
    std::shared_ptr<Layer> k_norm_;
    std::shared_ptr<Layer> post_attention_norm_;
    std::shared_ptr<Layer> gate_proj_;
    std::shared_ptr<Layer> up_proj_;
    std::shared_ptr<Layer> down_proj_;
};

struct Qwen3Layers
{
    std::shared_ptr<Layer> embedding_layer_;
    std::vector<Qwen3DecoderLayer> decoder_layers_;
    std::shared_ptr<Layer> final_norm_;
    std::shared_ptr<Layer> lm_head_;

    // Operators shared by decoder layers.
    std::shared_ptr<Layer> add_layer_;
    std::shared_ptr<Layer> swiglu_layer_;
    bool tied_weights_ = false;

    void to_cuda(const std::shared_ptr<CudaConfig>& config);
};

class Qwen3Model : public Model
{
public:
    Qwen3Model(std::string tokenizer_path, std::string model_path,
               int32_t max_seq_len = 2048);
    ~Qwen3Model() override;

    Status init(DeviceType device_type) override;
    Status estimate_paged_kv_cache_blocks(int32_t block_size, int32_t max_batch_tokens,
                                          float memory_utilization, int32_t& num_blocks) const;
    size_t paged_kv_cache_size_bytes(int32_t num_blocks, int32_t block_size) const;
    Status configure_paged_kv_cache(int32_t num_blocks, int32_t block_size,
                                    int32_t max_batch_tokens);
    Status forward_paged_batch(const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& positions,
                               const std::vector<std::vector<int32_t>>& block_tables,
                               const std::vector<size_t>& sample_rows,
                               std::vector<std::vector<float>>& logits);
    std::vector<int32_t> tokenize_prompt(const std::string& prompt) const;
    std::string decode_tokens(const std::vector<int32_t>& token_ids) const;

    bool is_sentence_ending(int32_t token_idx) const override;
    std::string decode(int32_t token_idx) const override;
    std::string decode(std::vector<int32_t> token_idxs) const override;

protected:
    Status read_model_file() override;
    Status create_encode_layer() override;
    Status generate_model_infos(const ModelConfig& config) const override;
    Status create_layers() override;

private:
    struct SafeTensorInfo
    {
        std::vector<int32_t> shape;
        size_t begin = 0;
        size_t end = 0;
        std::string dtype;
    };

    std::vector<int32_t> encode(const std::string& sentence) const override;
    void create_param_layers();
    void create_shared_layers();
    void init_mem();

    Status embedding(const std::vector<int32_t>& tokens, EmbeddingOutput& output) const;
    float* load_weight(const std::string& name, const std::vector<int32_t>& expected_shape);
    Status read_qwen3_config(ModelConfig& config);
    Status map_safetensors_file();
    void release_safetensors_map();

    int32_t requested_seq_len_ = 2048;
    int32_t qwen_head_dim_ = 0;
    float rms_norm_eps_ = 1e-6f;
    float rope_theta_ = 1000000.0f;

    std::shared_ptr<CudaConfig> cuda_config_;
    std::unique_ptr<Qwen3Layers> qwen3_layers_;
    int32_t paged_num_blocks_ = 0;
    int32_t paged_block_size_ = 0;
    int32_t paged_max_batch_tokens_ = 0;
    int32_t paged_max_table_entries_ = 0;

    int safetensors_fd_ = -1;
    void* safetensors_mapping_ = nullptr;
    size_t safetensors_file_size_ = 0;
    size_t safetensors_data_start_ = 0;
    std::unordered_map<std::string, SafeTensorInfo> safetensors_;
    std::vector<std::unique_ptr<float[]>> weight_storage_;
    std::string weight_loading_error_;
};

}  // namespace my_vllm

#endif  // MYVLLM_MODEL_QWEN3_H_
