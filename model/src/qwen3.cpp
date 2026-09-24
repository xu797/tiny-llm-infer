#include <algorithm>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <nlohmann/json.hpp>

#include "qwen3.h"
#include "argmax_sampler.h"
#include "cuda_alloc.h"
#include "cpu_alloc.h"
#include "matmul.h"
#include "mha.h"
#include "rmsnorm.h"
#include "rope_kernel_cpu.h"
#include "rope_kernel_cuda.cuh"
#include "swiglu.h"

namespace my_vllm
{
namespace
{
bool checked_element_count(const std::vector<int32_t>& dims, size_t& count)
{
    count = 1;
    for (const int32_t dim : dims)
    {
        if (dim <= 0 || count > std::numeric_limits<size_t>::max() / static_cast<size_t>(dim))
        {
            return false;
        }
        count *= static_cast<size_t>(dim);
    }
    return true;
}
}  // namespace

Qwen3Model::Qwen3Model(std::string tokenizer_path, std::string model_path,
                       int32_t max_seq_len)
    : LLama2Model(TokenizerType::kEncodeBpe, std::move(tokenizer_path),
                  std::move(model_path), false),
      requested_seq_len_(max_seq_len)
{
    model_type_ = ModelType::kModelTypeQwen3;
}

Qwen3Model::~Qwen3Model()
{
    release_safetensors_map();
}

Status Qwen3Model::init(DeviceType device_type)
{
    if (token_path_.empty() || model_path_.empty())
    {
        return PathNotValid("Both --tokenizer and --model paths are required for Qwen3.");
    }
    if (requested_seq_len_ <= 0)
    {
        return InvalidArgument("The Qwen3 context length must be positive.");
    }
    if (device_type != DeviceType::kDeviceCPU && device_type != DeviceType::kDeviceCUDA)
    {
        return InvalidArgument("The requested device type is not supported.");
    }

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA)
    {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess)
        {
            return InternalError(std::string("Failed to select CUDA device 0: ") +
                                 cudaGetErrorString(error));
        }
        cuda_config_ = std::make_shared<CudaConfig>();
        error = cudaStreamCreate(&cuda_config_->stream);
        if (error != cudaSuccess)
        {
            return InternalError(std::string("Failed to create CUDA stream: ") +
                                 cudaGetErrorString(error));
        }
    }

    Status status = gen_model_from_file();
    if (!status) return status;

    init_mem();
    if (device_type_ == DeviceType::kDeviceCPU)
    {
        qwen3_sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_, rope_theta_,
                                     get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                     get_buffer(ModelBufferType::kCosCache).ptr<float>());
    }
    else
    {
        qwen3_sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_, rope_theta_,
                                    get_buffer(ModelBufferType::kSinCache),
                                    get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
        const cudaError_t error = cudaStreamSynchronize(cuda_config_->stream);
        if (error != cudaSuccess)
        {
            return InternalError(std::string("Failed to upload Qwen3 weights or initialize RoPE: ") +
                                 cudaGetErrorString(error));
        }
        // CUDA layers own device copies now; discard the temporary FP32 staging
        // arrays so the 0.6B model does not consume another 2.4 GB of host RAM.
        weight_storage_.clear();
        weight_storage_.shrink_to_fit();
    }

    sampler_ = std::make_unique<ArgmaxSampler>(device_type_);
    return Success();
}

Status Qwen3Model::read_model_file()
{
    using json = nlohmann::json;
    const std::filesystem::path model_file(model_path_);
    const std::filesystem::path config_file = model_file.parent_path() / "config.json";
    std::ifstream config_input(config_file);
    if (!config_input)
    {
        return PathNotValid("Cannot open the Qwen3 config file: " + config_file.string());
    }

    json model_config;
    try
    {
        model_config = json::parse(config_input);
        if (model_config.value("model_type", std::string()) != "qwen3")
        {
            return ModelParseError("The config.json next to the weights is not a Qwen3 model.");
        }
        qwen_head_dim_ = model_config.value("head_dim", 0);
        rms_norm_eps_ = model_config.value("rms_norm_eps", 1e-6f);
        rope_theta_ = model_config.value("rope_theta", 1000000.0f);
    }
    catch (const std::exception& error)
    {
        return ModelParseError(std::string("Failed to parse Qwen3 config.json: ") + error.what());
    }

    safetensors_fd_ = open(model_path_.c_str(), O_RDONLY);
    if (safetensors_fd_ < 0)
    {
        return PathNotValid("Cannot open the Qwen3 safetensors file: " + model_path_);
    }
    struct stat file_stat
    {
    };
    if (fstat(safetensors_fd_, &file_stat) != 0 || file_stat.st_size < 8)
    {
        release_safetensors_map();
        return ModelParseError("The Qwen3 safetensors file is too small or cannot be read.");
    }
    safetensors_file_size_ = static_cast<size_t>(file_stat.st_size);
    safetensors_mapping_ = mmap(nullptr, safetensors_file_size_, PROT_READ, MAP_PRIVATE,
                                safetensors_fd_, 0);
    if (safetensors_mapping_ == MAP_FAILED || safetensors_mapping_ == nullptr)
    {
        safetensors_mapping_ = nullptr;
        release_safetensors_map();
        return ModelParseError("Failed to map the Qwen3 safetensors file into memory.");
    }

    uint64_t header_size = 0;
    std::memcpy(&header_size, safetensors_mapping_, sizeof(header_size));
    if (header_size > safetensors_file_size_ - sizeof(header_size))
    {
        release_safetensors_map();
        return ModelParseError("The Qwen3 safetensors header size is invalid.");
    }
    safetensors_data_start_ = sizeof(header_size) + static_cast<size_t>(header_size);

    try
    {
        const char* header_data = static_cast<const char*>(safetensors_mapping_) + sizeof(header_size);
        const json header = json::parse(std::string(header_data, static_cast<size_t>(header_size)));
        const size_t data_size = safetensors_file_size_ - safetensors_data_start_;
        for (auto item = header.begin(); item != header.end(); ++item)
        {
            if (item.key() == "__metadata__") continue;
            SafeTensorInfo info;
            info.dtype = item.value().at("dtype").get<std::string>();
            for (const auto& dim : item.value().at("shape"))
            {
                info.shape.push_back(dim.get<int32_t>());
            }
            const auto offsets = item.value().at("data_offsets");
            info.begin = offsets.at(0).get<size_t>();
            info.end = offsets.at(1).get<size_t>();
            if (info.begin > info.end || info.end > data_size)
            {
                release_safetensors_map();
                return ModelParseError("A Qwen3 tensor has an invalid safetensors data offset.");
            }
            safetensors_.emplace(item.key(), std::move(info));
        }
    }
    catch (const std::exception& error)
    {
        release_safetensors_map();
        return ModelParseError(std::string("Failed to parse Qwen3 safetensors header: ") + error.what());
    }

    ModelConfig config;
    try
    {
        config.dim = model_config.at("hidden_size").get<int32_t>();
        config.hidden_dim = model_config.at("intermediate_size").get<int32_t>();
        config.layer_num = model_config.at("num_hidden_layers").get<int32_t>();
        config.head_num = model_config.at("num_attention_heads").get<int32_t>();
        config.kv_head_num = model_config.at("num_key_value_heads").get<int32_t>();
        config.vocab_size = model_config.at("vocab_size").get<int32_t>();
        const int32_t model_context = model_config.at("max_position_embeddings").get<int32_t>();
        config.seq_len = std::min(requested_seq_len_, model_context);
    }
    catch (const std::exception& error)
    {
        release_safetensors_map();
        return ModelParseError(std::string("Qwen3 config.json is missing a model dimension: ") + error.what());
    }
    const Status status = generate_model_infos(config);
    if (!status) release_safetensors_map();
    return status;
}

Status Qwen3Model::generate_model_infos(const ModelConfig& config) const
{
    if (qwen_head_dim_ <= 0 || config.head_num <= 0 || config.kv_head_num <= 0 ||
        config.vocab_size <= 0 || config.seq_len <= 0 || config.kv_head_num > config.head_num ||
        config.head_num % config.kv_head_num != 0 || qwen_head_dim_ % 2 != 0)
    {
        return ModelParseError("The Qwen3 config.json contains invalid attention dimensions.");
    }
    Status status = Model::generate_model_infos(config);
    if (!status) return status;
    config_->head_size_ = qwen_head_dim_;
    config_->kv_dim_ = config.kv_head_num * qwen_head_dim_;
    config_->kv_mul_ = config.head_num / config.kv_head_num;
    config_->query_dim_ = config.head_num * qwen_head_dim_;
    return Success();
}

Status Qwen3Model::create_encode_layer()
{
    try
    {
        encode_layer_ = std::make_unique<Qwen3EncodeLayer>(token_path_, false, false);
    }
    catch (const std::exception& error)
    {
        return ModelParseError(std::string("Failed to create Qwen3 tokenizer: ") + error.what());
    }
    if (!encode_layer_ || encode_layer_->vocab_size() <= 0)
    {
        return InternalError("The Qwen3 tokenizer vocabulary is empty.");
    }
    return Success();
}

float* Qwen3Model::load_weight(const std::string& name,
                               const std::vector<int32_t>& expected_shape)
{
    const auto found = safetensors_.find(name);
    if (found == safetensors_.end())
    {
        weight_loading_error_ = "Missing Qwen3 tensor: " + name;
        return nullptr;
    }
    const SafeTensorInfo& info = found->second;
    if (info.shape != expected_shape)
    {
        weight_loading_error_ = "Unexpected shape for Qwen3 tensor " + name;
        return nullptr;
    }
    size_t count = 0;
    if (!checked_element_count(expected_shape, count))
    {
        weight_loading_error_ = "Invalid element count for Qwen3 tensor " + name;
        return nullptr;
    }
    const size_t element_size = info.dtype == "BF16" ? 2 : (info.dtype == "F32" ? 4 : 0);
    if (element_size == 0 || info.end - info.begin != count * element_size)
    {
        weight_loading_error_ = "Unsupported dtype or byte size for Qwen3 tensor " + name +
                                " (dtype=" + info.dtype + ")";
        return nullptr;
    }

    std::unique_ptr<float[]> converted(new float[count]);
    const uint8_t* source = static_cast<const uint8_t*>(safetensors_mapping_) +
                            safetensors_data_start_ + info.begin;
    if (info.dtype == "BF16")
    {
        for (size_t i = 0; i < count; ++i)
        {
            uint16_t bf16;
            std::memcpy(&bf16, source + i * sizeof(uint16_t), sizeof(uint16_t));
            const uint32_t bits = static_cast<uint32_t>(bf16) << 16;
            std::memcpy(converted.get() + i, &bits, sizeof(float));
        }
    }
    else
    {
        std::memcpy(converted.get(), source, count * sizeof(float));
    }

    float* result = converted.get();
    weight_storage_.push_back(std::move(converted));
    return result;
}

void Qwen3Model::create_param_layers()
{
    CHECK(llama_layers_ != nullptr);
    weight_loading_error_.clear();
    const int32_t dim = config_->dim_;
    const int32_t query_dim = config_->query_dim_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    const int32_t vocab_size = config_->vocab_size_;
    const DeviceType cpu = DeviceType::kDeviceCPU;

    auto embedding_weight = load_weight("model.embed_tokens.weight", {vocab_size, dim});
    if (!embedding_weight) return;
    auto embedding = std::make_shared<EmbeddingLayer>(device_type_, dim, config_->seq_len_, vocab_size);
    CHECK(embedding->set_weight(0, {vocab_size, dim}, embedding_weight, cpu));
    llama_layers_->embedding_layer_ = embedding;

    auto create_matmul = [this, cpu](const std::string& name, int32_t output_dim,
                                     int32_t input_dim) -> std::shared_ptr<Layer> {
        float* weight = load_weight(name, {output_dim, input_dim});
        if (!weight) return nullptr;
        auto layer = std::make_shared<MatmulLayer>(device_type_, output_dim, input_dim);
        CHECK(layer->set_weight(0, {output_dim, input_dim}, weight, cpu));
        return layer;
    };
    auto create_norm = [this, cpu](const std::string& name, int32_t width) -> std::shared_ptr<Layer> {
        float* weight = load_weight(name, {width});
        if (!weight) return nullptr;
        auto layer = std::make_shared<RmsNormLayer>(device_type_, width, rms_norm_eps_);
        CHECK(layer->set_weight(0, {width}, weight, cpu));
        return layer;
    };

    std::vector<std::shared_ptr<Layer>> attention_norms;
    std::vector<std::shared_ptr<Layer>> ffn_norms;
    attention_norms.reserve(config_->layer_num_);
    ffn_norms.reserve(config_->layer_num_);
    for (int32_t i = 0; i < config_->layer_num_; ++i)
    {
        const std::string prefix = "model.layers." + std::to_string(i) + ".";
        auto input_norm = create_norm(prefix + "input_layernorm.weight", dim);
        auto post_attention_norm = create_norm(prefix + "post_attention_layernorm.weight", dim);
        auto q_norm = create_norm(prefix + "self_attn.q_norm.weight", qwen_head_dim_);
        auto k_norm = create_norm(prefix + "self_attn.k_norm.weight", qwen_head_dim_);
        auto q = create_matmul(prefix + "self_attn.q_proj.weight", query_dim, dim);
        auto k = create_matmul(prefix + "self_attn.k_proj.weight", kv_dim, dim);
        auto v = create_matmul(prefix + "self_attn.v_proj.weight", kv_dim, dim);
        auto o = create_matmul(prefix + "self_attn.o_proj.weight", dim, query_dim);
        auto gate = create_matmul(prefix + "mlp.gate_proj.weight", hidden_dim, dim);
        auto up = create_matmul(prefix + "mlp.up_proj.weight", hidden_dim, dim);
        auto down = create_matmul(prefix + "mlp.down_proj.weight", dim, hidden_dim);
        if (!input_norm || !post_attention_norm || !q_norm || !k_norm || !q || !k || !v || !o ||
            !gate || !up || !down)
        {
            return;
        }
        attention_norms.push_back(input_norm);
        ffn_norms.push_back(post_attention_norm);
        llama_layers_->qnorm_layers_.push_back(q_norm);
        llama_layers_->knorm_layers_.push_back(k_norm);
        llama_layers_->wq_layers_.push_back(q);
        llama_layers_->wk_layers_.push_back(k);
        llama_layers_->wv_layers_.push_back(v);
        llama_layers_->wo_layers_.push_back(o);
        llama_layers_->w1_layers_.push_back(gate);
        llama_layers_->w3_layers_.push_back(up);
        llama_layers_->w2_layers_.push_back(down);
    }

    llama_layers_->rmsnorm_layers_.insert(llama_layers_->rmsnorm_layers_.end(),
                                          attention_norms.begin(), attention_norms.end());
    llama_layers_->rmsnorm_layers_.insert(llama_layers_->rmsnorm_layers_.end(),
                                          ffn_norms.begin(), ffn_norms.end());
    auto final_norm = create_norm("model.norm.weight", dim);
    if (!final_norm) return;
    llama_layers_->rmsnorm_layers_.push_back(final_norm);

    auto lm_head = std::make_shared<MatmulLayer>(device_type_, vocab_size, dim);
    CHECK(lm_head->set_weight(0, {vocab_size, dim}, embedding_weight, cpu));
    llama_layers_->cls_layer_ = lm_head;
    llama_layers_->tied_weights_ = true;
}

void Qwen3Model::create_nonparam_layers()
{
    llama_layers_->mha_layer_ = std::make_shared<MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_,
        config_->head_num_, config_->head_size_);
    llama_layers_->add_layer_ = std::make_shared<VecAddLayer>(device_type_);
    llama_layers_->swiglu_layer_ = std::make_shared<SwiGLULayer>(device_type_, config_->hidden_dim_);
}

Status Qwen3Model::create_layers()
{
    llama_layers_ = std::make_unique<LLama2Layers>();
    create_param_layers();
    if (!weight_loading_error_.empty())
    {
        release_safetensors_map();
        return ModelParseError(weight_loading_error_);
    }
    create_nonparam_layers();
    const size_t layer_count = static_cast<size_t>(config_->layer_num_);
    if (!llama_layers_->embedding_layer_ || !llama_layers_->cls_layer_ ||
        llama_layers_->wq_layers_.size() != layer_count ||
        llama_layers_->wk_layers_.size() != layer_count ||
        llama_layers_->wv_layers_.size() != layer_count ||
        llama_layers_->wo_layers_.size() != layer_count ||
        llama_layers_->w1_layers_.size() != layer_count ||
        llama_layers_->w2_layers_.size() != layer_count ||
        llama_layers_->w3_layers_.size() != layer_count ||
        llama_layers_->rmsnorm_layers_.size() != layer_count * 2 + 1 ||
        llama_layers_->qnorm_layers_.size() != layer_count ||
        llama_layers_->knorm_layers_.size() != layer_count)
    {
        release_safetensors_map();
        return InternalError("Failed to construct all Qwen3 transformer layers.");
    }
    release_safetensors_map();
    return Success();
}

void Qwen3Model::init_mem()
{
    LLama2Model::init_mem();
}

Status Qwen3Model::forward(const Tensor& input, const Tensor& pos_tensor, int& next) const
{
    UNUSED(next);
    if (input.is_empty() || !config_ || input.data_type() != DataType::kDataTypeFp32 ||
        input.device_type() != device_type_ || input.size() != static_cast<size_t>(config_->dim_))
    {
        return InvalidArgument("Qwen3 input must be one fp32 hidden-state vector on the model device.");
    }
    if (pos_tensor.is_empty() || pos_tensor.data_type() != DataType::kDataTypeInt32 ||
        pos_tensor.device_type() != DeviceType::kDeviceCPU || pos_tensor.size() != 1)
    {
        return InvalidArgument("Qwen3 position must contain one CPU int32 value.");
    }
    const int32_t pos = pos_tensor.ptr<int32_t>()[0];
    if (pos < 0 || pos >= config_->seq_len_)
    {
        return InvalidArgument("Qwen3 input position is outside the configured context window.");
    }

    for (int32_t layer = 0; layer < config_->layer_num_; ++layer)
    {
        attention_rms(layer, input);
        Tensor query = get_buffer(ModelBufferType::kQuery);
        const auto [key, value] = slice_kv_cache(layer, pos);
        const Tensor normalized = get_buffer(ModelBufferType::kOutputRMSNorm);
        STATUS_CHECK(llama_layers_->wq_layers_.at(layer)->forward(normalized, query));
        STATUS_CHECK(llama_layers_->wk_layers_.at(layer)->forward(normalized, key));
        STATUS_CHECK(llama_layers_->wv_layers_.at(layer)->forward(normalized, value));

        Tensor query_heads(DataType::kDataTypeFp32, config_->head_num_, config_->head_size_,
                           false, nullptr, query.ptr<float>());
        Tensor key_heads(DataType::kDataTypeFp32, config_->kv_head_num_, config_->head_size_,
                         false, nullptr, const_cast<float*>(key.ptr<float>()));
        query_heads.set_device_type(device_type_);
        key_heads.set_device_type(device_type_);
        STATUS_CHECK(llama_layers_->qnorm_layers_.at(layer)->forward(query_heads, query_heads));
        STATUS_CHECK(llama_layers_->knorm_layers_.at(layer)->forward(key_heads, key_heads));

        if (device_type_ == DeviceType::kDeviceCPU)
        {
            qwen3_rope_kernel_cpu(config_->head_num_, config_->kv_head_num_, config_->head_size_,
                                  query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), nullptr);
        }
        else
        {
            qwen3_rope_kernel_cu(config_->head_num_, config_->kv_head_num_, config_->head_size_,
                                 query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
                                 get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
        }

        attention_mha(layer, pos_tensor);
        feed_forward(layer, input);
    }
    cls_logits(input);
    return Success();
}

int32_t Qwen3Model::post_processing(const Tensor& pos, bool is_prompt) const
{
    UNUSED(pos);
    if (is_prompt) return -1;
    const Tensor& logits = get_buffer(ModelBufferType::kForwardOutput);
    return static_cast<int32_t>(sampler_->sample(
        logits.ptr<float>(), static_cast<size_t>(encode_layer_->vocab_size()),
        cuda_config_ ? cuda_config_->stream : nullptr));
}

void Qwen3Model::release_safetensors_map()
{
    if (safetensors_mapping_ != nullptr && safetensors_mapping_ != MAP_FAILED)
    {
        munmap(safetensors_mapping_, safetensors_file_size_);
        safetensors_mapping_ = nullptr;
    }
    if (safetensors_fd_ >= 0)
    {
        close(safetensors_fd_);
        safetensors_fd_ = -1;
    }
    safetensors_file_size_ = 0;
    safetensors_data_start_ = 0;
    safetensors_.clear();
}

}  // namespace my_vllm
