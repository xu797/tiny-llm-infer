#include "qwen3.h"

#include <cstring>
#include <limits>

#include <glog/logging.h>

#include "add.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "matmul.h"
#include "rmsnorm.h"
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
            return false;
        count *= static_cast<size_t>(dim);
    }
    return true;
}

template <typename... Layers>
void move_layers_to_cuda(const std::shared_ptr<CudaConfig>& config, Layers&... layers)
{
    const auto move_one = [&config](const std::shared_ptr<Layer>& layer) {
        if (!layer) return;
        layer->set_cuda_config(config);
        layer->to_cuda();
    };
    (move_one(layers), ...);
}
}  // namespace

void Qwen3Layers::to_cuda(const std::shared_ptr<CudaConfig>& config)
{
    add_layer_->set_cuda_config(config);
    swiglu_layer_->set_cuda_config(config);

    if (tied_weights_)
        lm_head_->set_cuda_config(config);
    else
        move_layers_to_cuda(config, lm_head_);

    move_layers_to_cuda(config, embedding_layer_, final_norm_);
    add_layer_->to_cuda();
    swiglu_layer_->to_cuda();

    // Move each decoder layer as a unit, matching the checkpoint/model layout.
    for (Qwen3DecoderLayer& layer : decoder_layers_)
    {
        move_layers_to_cuda(config, layer.input_norm_, layer.q_proj_, layer.k_proj_,
                            layer.v_proj_, layer.o_proj_, layer.q_norm_, layer.k_norm_,
                            layer.post_attention_norm_, layer.gate_proj_, layer.up_proj_,
                            layer.down_proj_);
    }

    if (tied_weights_)
    {
        auto lm_head = std::dynamic_pointer_cast<LayerParam>(lm_head_);
        auto embedding = std::dynamic_pointer_cast<LayerParam>(embedding_layer_);
        CHECK(lm_head != nullptr && embedding != nullptr);
        CHECK(lm_head->set_weight(0, embedding->get_weight(0)));
    }
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
    if (element_size == 0 || info.end < info.begin ||
        info.end - info.begin != count * element_size)
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
    const DeviceType host = DeviceType::kDeviceCPU;

    const auto create_projection = [this, host](const std::string& name,
                                                int32_t output_dim,
                                                int32_t input_dim) -> std::shared_ptr<Layer> {
        float* weights = load_weight(name, {output_dim, input_dim});
        if (!weights) return nullptr;
        auto layer = std::make_shared<MatmulLayer>(device_type_, output_dim, input_dim);
        CHECK(layer->set_weight(0, {output_dim, input_dim}, weights, host));
        return layer;
    };
    const auto create_norm = [this, host](const std::string& name,
                                          int32_t width) -> std::shared_ptr<Layer> {
        float* weights = load_weight(name, {width});
        if (!weights) return nullptr;
        auto layer = std::make_shared<RmsNormLayer>(device_type_, width, rms_norm_eps_);
        CHECK(layer->set_weight(0, {width}, weights, host));
        return layer;
    };

    float* embedding_weights = load_weight("model.embed_tokens.weight", {vocab_size, dim});
    if (!embedding_weights) return;
    auto embedding = std::make_shared<EmbeddingLayer>(device_type_, dim,
                                                       config_->seq_len_, vocab_size);
    CHECK(embedding->set_weight(0, {vocab_size, dim}, embedding_weights, host));
    qwen3_layers_->embedding_layer_ = embedding;

    qwen3_layers_->decoder_layers_.reserve(static_cast<size_t>(config_->layer_num_));
    for (int32_t index = 0; index < config_->layer_num_; ++index)
    {
        const std::string prefix = "model.layers." + std::to_string(index) + ".";
        Qwen3DecoderLayer layer;
        layer.input_norm_ = create_norm(prefix + "input_layernorm.weight", dim);
        layer.q_proj_ = create_projection(prefix + "self_attn.q_proj.weight", query_dim, dim);
        layer.k_proj_ = create_projection(prefix + "self_attn.k_proj.weight", kv_dim, dim);
        layer.v_proj_ = create_projection(prefix + "self_attn.v_proj.weight", kv_dim, dim);
        layer.o_proj_ = create_projection(prefix + "self_attn.o_proj.weight", dim, query_dim);
        layer.q_norm_ = create_norm(prefix + "self_attn.q_norm.weight", qwen_head_dim_);
        layer.k_norm_ = create_norm(prefix + "self_attn.k_norm.weight", qwen_head_dim_);
        layer.post_attention_norm_ =
            create_norm(prefix + "post_attention_layernorm.weight", dim);
        layer.gate_proj_ = create_projection(prefix + "mlp.gate_proj.weight", hidden_dim, dim);
        layer.up_proj_ = create_projection(prefix + "mlp.up_proj.weight", hidden_dim, dim);
        layer.down_proj_ = create_projection(prefix + "mlp.down_proj.weight", dim, hidden_dim);

        if (!layer.input_norm_ || !layer.q_proj_ || !layer.k_proj_ || !layer.v_proj_ ||
            !layer.o_proj_ || !layer.q_norm_ || !layer.k_norm_ ||
            !layer.post_attention_norm_ || !layer.gate_proj_ || !layer.up_proj_ ||
            !layer.down_proj_)
            return;
        qwen3_layers_->decoder_layers_.push_back(std::move(layer));
    }

    qwen3_layers_->final_norm_ = create_norm("model.norm.weight", dim);
    if (!qwen3_layers_->final_norm_) return;

    auto lm_head = std::make_shared<MatmulLayer>(device_type_, vocab_size, dim);
    CHECK(lm_head->set_weight(0, {vocab_size, dim}, embedding_weights, host));
    qwen3_layers_->lm_head_ = lm_head;
    qwen3_layers_->tied_weights_ = true;
}

void Qwen3Model::create_shared_layers()
{
    qwen3_layers_->add_layer_ = std::make_shared<VecAddLayer>(device_type_);
    qwen3_layers_->swiglu_layer_ =
        std::make_shared<SwiGLULayer>(device_type_, config_->hidden_dim_);
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

    create_shared_layers();
    const bool complete = qwen3_layers_->embedding_layer_ &&
                          qwen3_layers_->final_norm_ && qwen3_layers_->lm_head_ &&
                          qwen3_layers_->decoder_layers_.size() ==
                              static_cast<size_t>(config_->layer_num_);
    if (!complete)
    {
        release_safetensors_map();
        return InternalError("Failed to construct the Qwen3 model layers.");
    }

    release_safetensors_map();
    return Success();
}

void Qwen3Model::init_mem()
{
    const std::shared_ptr<DeviceAllocator> device_allocator =
        device_type_ == DeviceType::kDeviceCUDA
            ? std::static_pointer_cast<DeviceAllocator>(
                  CUDADeviceAllocatorFactory::get_instance())
            : std::static_pointer_cast<DeviceAllocator>(
                  CPUDeviceAllocatorFactory::get_instance());
    const auto host_allocator = CPUDeviceAllocatorFactory::get_instance();

    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        CHECK(cuda_config_ != nullptr);
        qwen3_layers_->to_cuda(cuda_config_);
    }

    Tensor input_tokens(DataType::kDataTypeInt32, 1, true, host_allocator);
    Tensor input_embeddings(DataType::kDataTypeFp32, 1, config_->dim_, true, device_allocator);
    Tensor sin_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                     true, device_allocator);
    Tensor cos_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_,
                     true, device_allocator);
    CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
    CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));
    CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
    CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));
}
}  // namespace my_vllm
