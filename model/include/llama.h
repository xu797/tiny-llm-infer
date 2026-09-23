#ifndef MYVLLM_MODEL_LLAMA_H_
#define MYVLLM_MODEL_LLAMA_H_

#include "cuda_config.h"
#include "model.h"
#include "add.h"
#include "embedding.h"
#include "rope.h"
#include "swiglu.h"

namespace my_vllm
{

struct LLama2Layers
{
  std::shared_ptr<Layer> add_layer_;
  std::shared_ptr<Layer> rope_layer_;
  std::shared_ptr<Layer> swiglu_layer_;
  std::shared_ptr<Layer> mha_layer_;

  std::vector<std::shared_ptr<Layer>> wq_layers_;
  std::vector<std::shared_ptr<Layer>> wk_layers_;
  std::vector<std::shared_ptr<Layer>> wv_layers_;
  std::vector<std::shared_ptr<Layer>> wo_layers_;

  std::vector<std::shared_ptr<Layer>> w1_layers_;
  std::vector<std::shared_ptr<Layer>> w2_layers_;
  std::vector<std::shared_ptr<Layer>> rmsnorm_layers_;
  std::vector<std::shared_ptr<Layer>> w3_layers_;
  std::shared_ptr<Layer> cls_layer_;

  std::shared_ptr<Layer> embedding_layer_;

  void to_cuda(std::shared_ptr<CudaConfig> config);
};

class LLama2Model : public Model
{
public:
    explicit LLama2Model(TokenizerType tokenizer_type, std::string token_path, std::string model_path, bool is_quant_model);

    Status init(DeviceType device_type) override;

    Status predict(const Tensor& input, const Tensor& pos_tensor, bool is_prompt, int& next) const override;

    Status forward(const Tensor& input, const Tensor& pos_tensor, int& next) const override;

    std::vector<int32_t> encode(const std::string& sentence) const override;

    std::string decode(int32_t token_idx) const override;

    std::string decode(std::vector<int32_t> token_idxs) const override;

    std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx, int32_t token_pos) const override;

    bool is_sentence_ending(int32_t token_idx) const override;

    EmbeddingOutput embedding(const std::vector<int>& tokens) const;

    Tensor fill_input(const Tensor& pos_tensor, const EmbeddingOutput& embedding_output, bool is_prompt) const;

// private:
    void init_mem() override;

    Status create_layers() override;

    void create_param_layers() override;

    void create_nonparam_layers() override;

    void create_param_quant_layers() override;

    void attention_mha(int32_t layer_idx, const Tensor& pos_tensor) const;

    void attention_rms(int32_t layer_idx, const Tensor& input) const;

    void feed_forward(int32_t layer_idx, const Tensor& input) const;

    void attention_qkv(int32_t layer_idx, const Tensor& pos_tensor) const;

    void cls_logits(const Tensor& input) const;

    int32_t post_processing(const Tensor& pos, bool is_prompt) const override;

// private:
    std::shared_ptr<CudaConfig> cuda_config_;
    std::unique_ptr<LLama2Layers> llama_layers_;
};
}

#endif
