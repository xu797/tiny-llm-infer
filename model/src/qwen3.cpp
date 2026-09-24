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
#include "add.h"
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

void Qwen3Layers::to_cuda(const std::shared_ptr<CudaConfig>& config)
{
    auto move_layer = [&config](const std::shared_ptr<Layer>& layer, bool move_weights = true) {
        if (!layer) return;
        layer->set_cuda_config(config);
        if (move_weights) layer->to_cuda();
    };

    move_layer(add_layer_);
    move_layer(swiglu_layer_);
    move_layer(cls_layer_, !tied_weights_);
    move_layer(embedding_layer_);

    // lm_head.weight is tied to embed_tokens.weight. Reuse its CUDA buffer so
    // the large vocabulary matrix is uploaded only once.
    if (tied_weights_ && cls_layer_ && embedding_layer_)
    {
        auto cls = std::dynamic_pointer_cast<LayerParam>(cls_layer_);
        auto embedding = std::dynamic_pointer_cast<LayerParam>(embedding_layer_);
        CHECK(cls != nullptr && embedding != nullptr);
        CHECK(cls->set_weight(0, embedding->get_weight(0)));
        cls_layer_->set_cuda_config(config);
    }

    move_layer(mha_layer_);
    for (auto* layers : {&wq_layers_, &wk_layers_, &wv_layers_, &wo_layers_,
                         &w1_layers_, &w2_layers_, &w3_layers_, &rmsnorm_layers_,
                         &qnorm_layers_, &knorm_layers_})
    {
        for (const auto& layer : *layers) move_layer(layer);
    }
}

Qwen3Model::Qwen3Model(std::string tokenizer_path, std::string model_path,
                       int32_t max_seq_len)
    : Model(TokenizerType::kEncodeBpe, ModelType::kModelTypeQwen3,
            std::move(tokenizer_path), std::move(model_path), false),
      requested_seq_len_(max_seq_len)
{
}

Qwen3Model::~Qwen3Model()
{
    if (cuda_config_ && cuda_config_->stream)
    {
        cudaStreamSynchronize(cuda_config_->stream);
    }
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
    CHECK(qwen3_layers_ != nullptr);
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
    qwen3_layers_->embedding_layer_ = embedding;

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
        qwen3_layers_->qnorm_layers_.push_back(q_norm);
        qwen3_layers_->knorm_layers_.push_back(k_norm);
        qwen3_layers_->wq_layers_.push_back(q);
        qwen3_layers_->wk_layers_.push_back(k);
        qwen3_layers_->wv_layers_.push_back(v);
        qwen3_layers_->wo_layers_.push_back(o);
        qwen3_layers_->w1_layers_.push_back(gate);
        qwen3_layers_->w3_layers_.push_back(up);
        qwen3_layers_->w2_layers_.push_back(down);
    }

    qwen3_layers_->rmsnorm_layers_.insert(qwen3_layers_->rmsnorm_layers_.end(),
                                          attention_norms.begin(), attention_norms.end());
    qwen3_layers_->rmsnorm_layers_.insert(qwen3_layers_->rmsnorm_layers_.end(),
                                          ffn_norms.begin(), ffn_norms.end());
    auto final_norm = create_norm("model.norm.weight", dim);
    if (!final_norm) return;
    qwen3_layers_->rmsnorm_layers_.push_back(final_norm);

    auto lm_head = std::make_shared<MatmulLayer>(device_type_, vocab_size, dim);
    CHECK(lm_head->set_weight(0, {vocab_size, dim}, embedding_weight, cpu));
    qwen3_layers_->cls_layer_ = lm_head;
    qwen3_layers_->tied_weights_ = true;
}

void Qwen3Model::create_nonparam_layers()
{
    qwen3_layers_->mha_layer_ = std::make_shared<MultiHeadAttention>(
        device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_,
        config_->head_num_, config_->head_size_);
    qwen3_layers_->add_layer_ = std::make_shared<VecAddLayer>(device_type_);
    qwen3_layers_->swiglu_layer_ = std::make_shared<SwiGLULayer>(device_type_, config_->hidden_dim_);
}

void Qwen3Model::create_param_quant_layers()
{
    // Qwen3 checkpoints are loaded from Safetensors and converted to FP32
    // staging tensors. The legacy packed-int8 model format is not applicable.
}

Status Qwen3Model::create_layers()
{
    qwen3_layers_ = std::make_unique<Qwen3Layers>();
    create_param_layers();
    if (!weight_loading_error_.empty())
    {
        release_safetensors_map();
        return ModelParseError(weight_loading_error_);
    }
    create_nonparam_layers();
    const size_t layer_count = static_cast<size_t>(config_->layer_num_);
    if (!qwen3_layers_->embedding_layer_ || !qwen3_layers_->cls_layer_ ||
        qwen3_layers_->wq_layers_.size() != layer_count ||
        qwen3_layers_->wk_layers_.size() != layer_count ||
        qwen3_layers_->wv_layers_.size() != layer_count ||
        qwen3_layers_->wo_layers_.size() != layer_count ||
        qwen3_layers_->w1_layers_.size() != layer_count ||
        qwen3_layers_->w2_layers_.size() != layer_count ||
        qwen3_layers_->w3_layers_.size() != layer_count ||
        qwen3_layers_->rmsnorm_layers_.size() != layer_count * 2 + 1 ||
        qwen3_layers_->qnorm_layers_.size() != layer_count ||
        qwen3_layers_->knorm_layers_.size() != layer_count)
    {
        release_safetensors_map();
        return InternalError("Failed to construct all Qwen3 transformer layers.");
    }
    release_safetensors_map();
    return Success();
}

void Qwen3Model::init_mem()
{
    std::shared_ptr<DeviceAllocator> alloc;
    if (device_type_ == DeviceType::kDeviceCPU)
    {
        alloc = CPUDeviceAllocatorFactory::get_instance();
    }
    else
    {
        alloc = CUDADeviceAllocatorFactory::get_instance();
        CHECK(cuda_config_ != nullptr);
        qwen3_layers_->to_cuda(cuda_config_);
    }

    const auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
    Tensor input_tokens(DataType::kDataTypeInt32, 1, true, alloc_cpu);
    Tensor input_embeddings(DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
    Tensor sin_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
    Tensor cos_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

    Tensor rms_output(DataType::kDataTypeFp32, config_->dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
    CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
    CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));
    Tensor mha_output(DataType::kDataTypeFp32, config_->query_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kOutputMHA, mha_output));

    Tensor w1_output(DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
    Tensor w3_output(DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
    CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

    Tensor key_cache(DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                     config_->kv_dim_, true, alloc);
    Tensor value_cache(DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                       config_->kv_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
    CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

    Tensor query(DataType::kDataTypeFp32, config_->query_dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kQuery, query));
    Tensor pos_tensor(DataType::kDataTypeInt32, 1, true, alloc_cpu);
    CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

    Tensor scores(DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc);
    Tensor attention_output(DataType::kDataTypeFp32, config_->dim_, true, alloc);
    CHECK(insert_buffer(ModelBufferType::kScoreStorage, scores));
    CHECK(insert_buffer(ModelBufferType::kAttnOutput, attention_output));

    Tensor forward_output(DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        Tensor forward_output_cpu(DataType::kDataTypeFp32, config_->vocab_size_, true, alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
    }
    CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
}

std::pair<Tensor, Tensor> Qwen3Model::slice_kv_cache(int32_t layer_idx,
                                                      int32_t token_pos) const
{
    const int64_t layer_offset = static_cast<int64_t>(layer_idx) * config_->seq_len_ * config_->kv_dim_;
    const int64_t cache_offset = layer_offset + static_cast<int64_t>(token_pos) * config_->kv_dim_;
    float* key_ptr = const_cast<float*>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
    float* value_ptr = const_cast<float*>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

    auto key_buffer = std::make_shared<Buffer>(config_->kv_dim_ * sizeof(float), nullptr, key_ptr, true);
    auto value_buffer = std::make_shared<Buffer>(config_->kv_dim_ * sizeof(float), nullptr, value_ptr, true);
    key_buffer->set_device_type(device_type_);
    value_buffer->set_device_type(device_type_);
    Tensor key(DataType::kDataTypeFp32, config_->kv_dim_);
    Tensor value(DataType::kDataTypeFp32, config_->kv_dim_);
    CHECK(key.assign(key_buffer));
    CHECK(value.assign(value_buffer));
    return {key, value};
}

Status Qwen3Model::embedding(const std::vector<int32_t>& tokens,
                             EmbeddingOutput& output) const
{
    if (tokens.empty() || tokens.size() > static_cast<size_t>(config_->seq_len_))
    {
        return InvalidArgument("The Qwen3 embedding token count is outside the context window.");
    }
    Tensor input_tokens = get_buffer(ModelBufferType::kInputTokens);
    Tensor input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
    if (input_tokens.size() != tokens.size())
    {
        input_tokens.reshape({static_cast<int32_t>(tokens.size())});
        input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
    }
    for (size_t i = 0; i < tokens.size(); ++i)
    {
        input_tokens.index<int32_t>(static_cast<int64_t>(i)) = tokens[i];
    }

    Tensor token_count(DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
    const Status status = qwen3_layers_->embedding_layer_->forward(input_tokens, token_count,
                                                                   input_embeddings);
    if (!status) return status;
    output = EmbeddingOutput(input_tokens, input_embeddings, token_count);
    return Success();
}

Tensor Qwen3Model::fill_input(const Tensor& pos_tensor,
                              const EmbeddingOutput& embedding_output,
                              bool is_prompt) const
{
    const int32_t index = is_prompt ? pos_tensor.index<int32_t>(0) : 0;
    const Tensor& embeddings = embedding_output.input_embeddings;
    CHECK_GE(index, 0);
    CHECK_LT(index, embeddings.get_dim(0));
    auto input_buffer = std::make_shared<Buffer>(config_->dim_ * sizeof(float), nullptr,
                                                 const_cast<float*>(embeddings.ptr<float>(static_cast<int64_t>(index) * config_->dim_)), true);
    input_buffer->set_device_type(device_type_);
    Tensor input(DataType::kDataTypeFp32, config_->dim_);
    CHECK(input.assign(input_buffer));
    return input;
}

Status Qwen3Model::attention_rms(int32_t layer_idx, const Tensor& input) const
{
    const Tensor output = get_buffer(ModelBufferType::kOutputRMSNorm);
    return qwen3_layers_->rmsnorm_layers_.at(layer_idx)->forward(input, output);
}

Status Qwen3Model::attention_mha(int32_t layer_idx, const Tensor& pos_tensor) const
{
    Tensor query = get_buffer(ModelBufferType::kQuery);
    Tensor scores = get_buffer(ModelBufferType::kScoreStorage);
    Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
    Tensor value_cache = get_buffer(ModelBufferType::kValueCache);
    Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
    auto mha = std::dynamic_pointer_cast<MultiHeadAttention>(qwen3_layers_->mha_layer_);
    if (!mha) return InternalError("The Qwen3 multi-head attention layer is missing.");
    const int32_t pos = pos_tensor.index<int32_t>(0);
    mha->set_pos(pos);
    mha->set_layer_idx(layer_idx);
    Status status = qwen3_layers_->mha_layer_->forward(query, scores, key_cache, value_cache,
                                                       mha_output);
    if (!status) return status;
    return qwen3_layers_->wo_layers_.at(layer_idx)->forward(
        mha_output, get_buffer(ModelBufferType::kAttnOutput));
}

Status Qwen3Model::feed_forward(int32_t layer_idx, const Tensor& input) const
{
    Status status = qwen3_layers_->add_layer_->forward(
        input, get_buffer(ModelBufferType::kAttnOutput), input);
    if (!status) return status;

    Tensor ffn_norm = get_buffer(ModelBufferType::kFFNRMSNorm);
    status = qwen3_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_)->forward(input,
                                                                                         ffn_norm);
    if (!status) return status;

    Tensor gate = get_buffer(ModelBufferType::kW1Output);
    Tensor up = get_buffer(ModelBufferType::kW3Output);
    status = qwen3_layers_->w1_layers_.at(layer_idx)->forward(ffn_norm, gate);
    if (!status) return status;
    status = qwen3_layers_->w3_layers_.at(layer_idx)->forward(ffn_norm, up);
    if (!status) return status;
    status = qwen3_layers_->swiglu_layer_->forward(gate, up, gate);
    if (!status) return status;

    Tensor down = get_buffer(ModelBufferType::kW2Output);
    status = qwen3_layers_->w2_layers_.at(layer_idx)->forward(gate, down);
    if (!status) return status;
    return qwen3_layers_->add_layer_->forward(input, down, input);
}

Status Qwen3Model::cls_logits(const Tensor& input) const
{
    const auto& final_norm = qwen3_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
    Status status = final_norm->forward(input, input);
    if (!status) return status;
    return qwen3_layers_->cls_layer_->forward(input, get_buffer(ModelBufferType::kForwardOutput));
}

Status Qwen3Model::predict(const Tensor& input, const Tensor& pos_tensor,
                           bool is_prompt, int& next) const
{
    Status status = forward(input, pos_tensor, next);
    if (!status) return status;
    next = post_processing(pos_tensor, is_prompt);
    return Success();
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
        Status status = attention_rms(layer, input);
        if (!status) return status;
        Tensor query = get_buffer(ModelBufferType::kQuery);
        const auto [key, value] = slice_kv_cache(layer, pos);
        const Tensor normalized = get_buffer(ModelBufferType::kOutputRMSNorm);
        status = qwen3_layers_->wq_layers_.at(layer)->forward(normalized, query);
        if (!status) return status;
        status = qwen3_layers_->wk_layers_.at(layer)->forward(normalized, key);
        if (!status) return status;
        status = qwen3_layers_->wv_layers_.at(layer)->forward(normalized, value);
        if (!status) return status;

        Tensor query_heads(DataType::kDataTypeFp32, config_->head_num_, config_->head_size_,
                           false, nullptr, query.ptr<float>());
        Tensor key_heads(DataType::kDataTypeFp32, config_->kv_head_num_, config_->head_size_,
                         false, nullptr, const_cast<float*>(key.ptr<float>()));
        query_heads.set_device_type(device_type_);
        key_heads.set_device_type(device_type_);
        status = qwen3_layers_->qnorm_layers_.at(layer)->forward(query_heads, query_heads);
        if (!status) return status;
        status = qwen3_layers_->knorm_layers_.at(layer)->forward(key_heads, key_heads);
        if (!status) return status;

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

        status = attention_mha(layer, pos_tensor);
        if (!status) return status;
        status = feed_forward(layer, input);
        if (!status) return status;
    }
    return cls_logits(input);
}

std::vector<int32_t> Qwen3Model::encode(const std::string& sentence) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->encode(sentence);
}

bool Qwen3Model::is_sentence_ending(int32_t token_idx) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->is_sentence_ending(token_idx);
}

std::string Qwen3Model::decode(int32_t token_idx) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_idx);
}

std::string Qwen3Model::decode(std::vector<int32_t> token_idxs) const
{
    CHECK(encode_layer_ != nullptr);
    return encode_layer_->decode(token_idxs);
}

Status Qwen3Model::generate(const std::string& prompt, int32_t max_new_tokens,
                            std::string& output)
{
    output.clear();
    if (!config_ || !encode_layer_ || !sampler_)
    {
        return InternalError("The Qwen3 model must be initialized before generation.");
    }
    if (max_new_tokens < 0)
    {
        return InvalidArgument("The number of generated tokens cannot be negative.");
    }
    if (max_new_tokens == 0) return Success();

    const std::vector<int32_t> token_ids = encode(prompt);
    if (token_ids.empty()) return InvalidArgument("The prompt did not produce any Qwen3 tokens.");
    if (token_ids.size() > static_cast<size_t>(config_->seq_len_))
    {
        return InvalidArgument("The prompt is longer than the Qwen3 context window.");
    }
    const int32_t max_context_tokens = config_->seq_len_ - static_cast<int32_t>(token_ids.size()) + 1;
    if (max_new_tokens > max_context_tokens)
    {
        return InvalidArgument("The requested generation length exceeds the Qwen3 context window.");
    }

    EmbeddingOutput prompt_embedding(Tensor{}, Tensor{}, Tensor{});
    Status status = embedding(token_ids, prompt_embedding);
    if (!status) return status;
    Tensor& pos_tensor = get_buffer(ModelBufferType::kInputPos);
    int next = -1;
    for (size_t i = 0; i < token_ids.size(); ++i)
    {
        pos_tensor.index<int32_t>(0) = static_cast<int32_t>(i);
        const Tensor input = fill_input(pos_tensor, prompt_embedding, true);
        const bool skip_sampling = i + 1 < token_ids.size();
        status = predict(input, pos_tensor, skip_sampling, next);
        if (!status) return status;
    }

    std::vector<int32_t> generated_tokens;
    generated_tokens.reserve(max_new_tokens);
    for (int32_t i = 0; i < max_new_tokens; ++i)
    {
        if (next < 0) return InternalError("Qwen3 did not produce a token after the prompt.");
        if (is_sentence_ending(next)) break;
        generated_tokens.push_back(next);
        if (i + 1 == max_new_tokens) break;

        EmbeddingOutput next_embedding(Tensor{}, Tensor{}, Tensor{});
        status = embedding({next}, next_embedding);
        if (!status) return status;
        pos_tensor.index<int32_t>(0) = static_cast<int32_t>(token_ids.size()) + i;
        const Tensor input = fill_input(pos_tensor, next_embedding, false);
        status = predict(input, pos_tensor, false, next);
        if (!status) return status;
    }
    output = decode(generated_tokens);
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
