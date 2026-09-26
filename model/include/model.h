#ifndef MYVLLM_MODEL_H_
#define MYVLLM_MODEL_H_

#include <map>
#include <string>

#include "config.h"
#include "encode.h"
#include "layer.h"
#include "tensor.h"
#include "status.h"

namespace my_vllm
{
class Model 
{
public:
    explicit Model(ModelType model_type, std::string token_path, std::string model_path);
    virtual ~Model() = default;

    virtual Status init(DeviceType device_type) = 0;

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

    virtual Status read_model_file() = 0;

    virtual Status create_encode_layer() = 0;

    Status gen_model_from_file();

    virtual Status generate_model_infos(const ModelConfig& config) const;

private:
    virtual void init_mem() = 0;

    virtual Status create_layers() = 0;

    virtual std::vector<int32_t> encode(const std::string& sentence) const = 0;

protected:
    std::unique_ptr<TransformerConfig> config_;

    std::string token_path_;
    std::string model_path_;
    std::unique_ptr<EncodeLayerBase> encode_layer_;
    std::map<ModelBufferType, Tensor> buffers_;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    ModelType model_type_ = ModelType::kModelTypeUnknown;
};

}


#endif
