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

// Qwen3 owns its layer set and inference path. It depends only on the shared
// Model/layer interfaces, so the Llama2 implementation can be removed without
// affecting Qwen3.
struct Qwen3Layers
{
    std::shared_ptr<Layer> add_layer_;
    std::shared_ptr<Layer> swiglu_layer_;
    std::shared_ptr<Layer> mha_layer_;

    std::vector<std::shared_ptr<Layer>> wq_layers_;
    std::vector<std::shared_ptr<Layer>> wk_layers_;
    std::vector<std::shared_ptr<Layer>> wv_layers_;
    std::vector<std::shared_ptr<Layer>> wo_layers_;
    std::vector<std::shared_ptr<Layer>> w1_layers_;
    std::vector<std::shared_ptr<Layer>> w2_layers_;
    std::vector<std::shared_ptr<Layer>> w3_layers_;
    std::vector<std::shared_ptr<Layer>> rmsnorm_layers_;
    std::vector<std::shared_ptr<Layer>> qnorm_layers_;
    std::vector<std::shared_ptr<Layer>> knorm_layers_;

    std::shared_ptr<Layer> cls_layer_;
    std::shared_ptr<Layer> embedding_layer_;
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
    Status predict(const Tensor& input, const Tensor& pos_tensor,
                   bool is_prompt, int& next) const override;
    Status forward(const Tensor& input, const Tensor& pos_tensor, int& next) const override;
    Status generate(const std::string& prompt, int32_t max_new_tokens,
                    std::string& output);

    Status configure_paged_kv_cache(int32_t num_blocks, int32_t block_size,
                                    int32_t max_batch_tokens);
    Status set_paged_block_table(const std::vector<int32_t>& block_table);
    Status forward_paged_token(int32_t token_id, int32_t position);
    Status forward_paged_batch(const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& positions,
                               const std::vector<std::vector<int32_t>>& block_tables,
                               const std::vector<size_t>& sample_rows,
                               std::vector<std::vector<float>>& logits);
    Status copy_logits_to_host(std::vector<float>& logits) const;
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
    std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx,
                                            int32_t token_pos) const override;
    void create_param_layers() override;
    void create_nonparam_layers() override;
    void create_param_quant_layers() override;
    void init_mem() override;
    int32_t post_processing(const Tensor& pos, bool is_prompt) const override;

    Status embedding(const std::vector<int32_t>& tokens, EmbeddingOutput& output) const;
    Tensor fill_input(const Tensor& pos_tensor, const EmbeddingOutput& embedding_output,
                      bool is_prompt) const;
    Status attention_rms(int32_t layer_idx, const Tensor& input) const;
    Status attention_mha(int32_t layer_idx, const Tensor& pos_tensor) const;
    Status attention_mha_paged(int32_t layer_idx, int32_t position) const;
    std::pair<Tensor, Tensor> slice_paged_kv_cache(int32_t layer_idx, int32_t position) const;
    Status forward_paged(const Tensor& input, const Tensor& pos_tensor) const;
    Status feed_forward(int32_t layer_idx, const Tensor& input) const;
    Status cls_logits(const Tensor& input) const;

    float* load_weight(const std::string& name, const std::vector<int32_t>& expected_shape);
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
    Tensor paged_block_table_host_;
    Tensor paged_block_table_device_;
    std::vector<int32_t> paged_block_table_ids_;

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
