#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

#include <cuda_runtime_api.h>

#include "qwen3.h"
#include "rope_kernel_cpu.h"
#include "rope_kernel_cuda.cuh"

namespace my_vllm
{
Qwen3Model::Qwen3Model(std::string tokenizer_path, std::string model_path,
                       int32_t max_seq_len)
    : Model(ModelType::kModelTypeQwen3,
            std::move(tokenizer_path), std::move(model_path)),
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
        return PathNotValid("Both --tokenizer and --model paths are required for Qwen3.");
    if (requested_seq_len_ <= 0)
        return InvalidArgument("The Qwen3 context length must be positive.");
    if (device_type != DeviceType::kDeviceCPU && device_type != DeviceType::kDeviceCUDA)
        return InvalidArgument("The requested device type is not supported.");

    device_type_ = device_type;
    if (device_type == DeviceType::kDeviceCUDA)
    {
        cudaError_t error = cudaSetDevice(0);
        if (error != cudaSuccess)
            return InternalError(std::string("Failed to select CUDA device 0: ") +
                                 cudaGetErrorString(error));
        cuda_config_ = std::make_shared<CudaConfig>();
        error = cudaStreamCreate(&cuda_config_->stream);
        if (error != cudaSuccess)
            return InternalError(std::string("Failed to create CUDA stream: ") +
                                 cudaGetErrorString(error));
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
            return InternalError(std::string("Failed to upload Qwen3 weights or initialize RoPE: ") +
                                 cudaGetErrorString(error));
        weight_storage_.clear();
        weight_storage_.shrink_to_fit();
    }

    return Success();
}

Status Qwen3Model::estimate_paged_kv_cache_blocks(
    int32_t block_size, int32_t max_batch_tokens, float memory_utilization,
    int32_t& num_blocks) const
{
    num_blocks = 0;
    if (!config_ || device_type_ != DeviceType::kDeviceCUDA)
        return InvalidArgument("Automatic KV sizing requires an initialized CUDA model.");
    if (block_size <= 0 || block_size > config_->seq_len_ || max_batch_tokens <= 0 ||
        !std::isfinite(memory_utilization) || memory_utilization <= 0.0f ||
        memory_utilization > 0.95f)
        return InvalidArgument("Invalid KV cache sizing parameters.");

    size_t free_bytes = 0;
    size_t total_bytes = 0;
    const cudaError_t error = cudaMemGetInfo(&free_bytes, &total_bytes);
    if (error != cudaSuccess)
        return InternalError(std::string("Failed to query CUDA memory: ") +
                             cudaGetErrorString(error));

    size_t workspace_floats_per_token =
        static_cast<size_t>(9) * config_->dim_ +
        static_cast<size_t>(2) * config_->query_dim_ +
        static_cast<size_t>(2) * config_->kv_dim_ +
        static_cast<size_t>(3) * config_->hidden_dim_;
    const size_t score_floats_per_token =
        static_cast<size_t>(config_->head_num_) * static_cast<size_t>(config_->seq_len_);
    if (workspace_floats_per_token >
        std::numeric_limits<size_t>::max() - score_floats_per_token)
        return InvalidArgument("The model workspace estimate overflows size_t.");
    workspace_floats_per_token += score_floats_per_token;
    if (workspace_floats_per_token >
        std::numeric_limits<size_t>::max() /
            static_cast<size_t>(max_batch_tokens) / sizeof(float))
        return InvalidArgument("The batch workspace estimate overflows size_t.");
    const size_t workspace_bytes = workspace_floats_per_token *
                                   static_cast<size_t>(max_batch_tokens) * sizeof(float);
    const size_t reserve_bytes =
        std::max(std::max(static_cast<size_t>(256ull << 20), total_bytes / 20),
                 workspace_bytes);
    const size_t target_bytes = static_cast<size_t>(
        static_cast<double>(total_bytes) * memory_utilization);
    const size_t budget_bytes = target_bytes > total_bytes - free_bytes
                                    ? target_bytes - (total_bytes - free_bytes)
                                    : 0;
    if (budget_bytes <= reserve_bytes)
        return InvalidArgument("Not enough free GPU memory remains for a KV cache pool.");

    const size_t bytes_per_block = paged_kv_cache_size_bytes(1, block_size);
    if (bytes_per_block == 0)
        return InvalidArgument("The KV cache block size overflows the supported dimensions.");
    const size_t count = (budget_bytes - reserve_bytes) / bytes_per_block;
    if (count == 0 || count > static_cast<size_t>(std::numeric_limits<int32_t>::max()))
        return InvalidArgument("The available GPU memory cannot hold a KV cache block.");
    num_blocks = static_cast<int32_t>(count);
    return Success();
}

size_t Qwen3Model::paged_kv_cache_size_bytes(int32_t num_blocks, int32_t block_size) const
{
    if (!config_ || num_blocks <= 0 || block_size <= 0 ||
        block_size > config_->seq_len_)
        return 0;
    size_t bytes = 2;  // separate K and V tensors
    for (const size_t factor : {static_cast<size_t>(config_->layer_num_),
                                static_cast<size_t>(num_blocks),
                                static_cast<size_t>(block_size),
                                static_cast<size_t>(config_->kv_dim_),
                                sizeof(float)})
    {
        if (factor != 0 && bytes > std::numeric_limits<size_t>::max() / factor) return 0;
        bytes *= factor;
    }
    return bytes;
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

std::vector<int32_t> Qwen3Model::tokenize_prompt(const std::string& prompt) const
{
    return encode(prompt);
}

std::string Qwen3Model::decode_tokens(const std::vector<int32_t>& token_ids) const
{
    return decode(token_ids);
}


}  // namespace my_vllm
