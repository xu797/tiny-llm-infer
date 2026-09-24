#ifndef MYVLLM_MODEL_QWEN3_H_
#define MYVLLM_MODEL_QWEN3_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "llama.h"

namespace my_vllm
{

class Qwen3Model : public LLama2Model
{
public:
    Qwen3Model(std::string tokenizer_path, std::string model_path,
               int32_t max_seq_len = 2048);
    ~Qwen3Model() override;

    Status init(DeviceType device_type) override;
    Status forward(const Tensor& input, const Tensor& pos_tensor, int& next) const override;

protected:
    Status read_model_file() override;
    Status create_encode_layer() override;
    Status generate_model_infos(const ModelConfig& config) const override;
    Status create_layers() override;
    void create_param_layers() override;
    void create_nonparam_layers() override;
    void init_mem() override;
    int32_t post_processing(const Tensor& pos, bool is_prompt) const override;

private:
    struct SafeTensorInfo
    {
        std::vector<int32_t> shape;
        size_t begin = 0;
        size_t end = 0;
        std::string dtype;
    };

    float* load_weight(const std::string& name, const std::vector<int32_t>& expected_shape);
    void release_safetensors_map();

    int32_t requested_seq_len_ = 2048;
    int32_t qwen_head_dim_ = 0;
    float rms_norm_eps_ = 1e-6f;
    float rope_theta_ = 1000000.0f;

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
