#ifndef MYVLLM_MODEL_H_
#define MYVLLM_MODEL_H_

#include <map>
#include <string>

#include "config.h"
#include "encode.h"
#include "layer.h"
#include "raw_model_data.h"
#include "argmax_sampler.h"
#include "sentencepiece_processor.h"
#include "tensor.h"
#include "status.h"

namespace my_vllm
{
class Model 
{
public:
    explicit Model(TokenizerType tokenizer_type, ModelType model_type, std::string token_path, std::string model_path, bool is_quant_model);
    virtual ~Model() = default;

    virtual Status init(DeviceType device_type) = 0;

    virtual Status predict(const Tensor& input, const Tensor& pos_tensor,
                                bool is_prompt, int& next) const = 0;

    virtual Status forward(const Tensor& input, const Tensor& pos_tensor,
                                int& next) const = 0;

    ModelType model_type() const;

    const std::string& token_path() const;

    const std::string& model_path() const;

    virtual Tensor& get_buffer(ModelBufferType buffer_idx);

    virtual const Tensor& get_buffer(ModelBufferType buffer_idx) const;

    virtual bool is_sentence_ending(int32_t token_idx) const = 0;

    virtual std::string decode(int32_t token_idx) const = 0;

    virtual std::string decode(std::vector<int32_t> token_idxs) const = 0;

protected:
    virtual Status insert_buffer(ModelBufferType buffer_idx, const Tensor& tensor);

    virtual Status read_model_file();

    virtual Status create_encode_layer();

    virtual Status gen_model_from_file();

    virtual Status generate_model_infos(const ModelConfig& config) const;

    virtual int32_t post_processing(const Tensor& pos, bool is_prompt) const = 0;

private:
    virtual void init_mem() = 0;

    virtual Status create_layers() = 0;

    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

    virtual std::pair<Tensor, Tensor> slice_kv_cache(int32_t layer_idx,
                                                                    int32_t token_pos) const = 0;

    virtual void create_param_layers() = 0;

    virtual void create_nonparam_layers() = 0;

    virtual void create_param_quant_layers() = 0;

protected:
    int32_t group_size_ = 1;
    bool is_quant_model_ = false;
    std::unique_ptr<TransformerConfig> config_;

    std::string token_path_;
    std::string model_path_;
    std::unique_ptr<EncodeLayerBase> encode_layer_;
    std::map<ModelBufferType, Tensor> buffers_;
    std::unique_ptr<Sampler> sampler_;
    std::shared_ptr<RawModelData> raw_model_data_;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    ModelType model_type_ = ModelType::kModelTypeUnknown;
    TokenizerType tokenizer_type_ = TokenizerType::kEncodeUnknown;
};

}


#endif
