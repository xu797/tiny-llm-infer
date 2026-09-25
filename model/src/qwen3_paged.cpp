#include "qwen3.h"

#include <algorithm>
#include <cuda_runtime_api.h>

#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "paged_attention.h"
#include "rope_kernel_cpu.h"
#include "rope_kernel_cuda.cuh"

namespace my_vllm
{
Status Qwen3Model::configure_paged_kv_cache(int32_t num_blocks, int32_t block_size,
                                               int32_t max_batch_tokens)
{
    if (not config_ or num_blocks <= 0 or block_size <= 0 or
        block_size > config_->seq_len_ or max_batch_tokens <= 0)
        return InvalidArgument("Invalid paged KV cache dimensions.");
    std::shared_ptr<DeviceAllocator> allocator;
    if (device_type_ == DeviceType::kDeviceCPU)
        allocator = CPUDeviceAllocatorFactory::get_instance();
    else if (device_type_ == DeviceType::kDeviceCUDA)
        allocator = CUDADeviceAllocatorFactory::get_instance();
    else
        return InvalidArgument("Initialize Qwen3 before configuring its KV cache.");

    buffers_[ModelBufferType::kKeyCache] =
        Tensor(DataType::kDataTypeFp32, config_->layer_num_, num_blocks,
               block_size, config_->kv_dim_, true, allocator);
    buffers_[ModelBufferType::kValueCache] =
        Tensor(DataType::kDataTypeFp32, config_->layer_num_, num_blocks,
               block_size, config_->kv_dim_, true, allocator);
    paged_num_blocks_ = num_blocks;
    paged_block_size_ = block_size;
    paged_max_batch_tokens_ = max_batch_tokens;
    paged_max_table_entries_ = (config_->seq_len_ + block_size - 1) / block_size;
    paged_block_table_host_ = Tensor(DataType::kDataTypeInt32, paged_max_table_entries_,
                                     true, CPUDeviceAllocatorFactory::get_instance());
    if (device_type_ == DeviceType::kDeviceCUDA)
        paged_block_table_device_ = Tensor(DataType::kDataTypeInt32,
                                           paged_max_table_entries_, true, allocator);
    else
        paged_block_table_device_ = paged_block_table_host_;
    paged_block_table_ids_.clear();
    return Success();
}

Status Qwen3Model::set_paged_block_table(const std::vector<int32_t>& block_table)
{
    if (paged_num_blocks_ <= 0 or block_table.empty() or
        block_table.size() > static_cast<size_t>(paged_max_table_entries_))
        return InvalidArgument("Invalid Qwen3 KV block table.");
    for (const int32_t block : block_table)
        if (block < 0 or block >= paged_num_blocks_)
            return InvalidArgument("KV block table contains an invalid block id.");
    if (block_table == paged_block_table_ids_) return Success();
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        const cudaError_t sync_error = cudaStreamSynchronize(cuda_config_->stream);
        if (sync_error != cudaSuccess)
            return InternalError(cudaGetErrorString(sync_error));
    }
    int32_t* host = paged_block_table_host_.ptr<int32_t>();
    std::fill(host, host + paged_max_table_entries_, -1);
    std::copy(block_table.begin(), block_table.end(), host);
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        const size_t bytes = static_cast<size_t>(paged_max_table_entries_) * sizeof(int32_t);
        const cudaError_t error = cudaMemcpyAsync(paged_block_table_device_.ptr<int32_t>(),
            host, bytes, cudaMemcpyHostToDevice, cuda_config_->stream);
        if (error != cudaSuccess) return InternalError(cudaGetErrorString(error));
    }
    paged_block_table_ids_ = block_table;
    return Success();
}

std::pair<Tensor, Tensor> Qwen3Model::slice_paged_kv_cache(int32_t layer, int32_t position) const
{
    const size_t logical = static_cast<size_t>(position / paged_block_size_);
    CHECK_LT(logical, paged_block_table_ids_.size());
    const int32_t physical = paged_block_table_ids_[logical];
    const size_t offset =
        ((static_cast<size_t>(layer) * paged_num_blocks_ + physical) * paged_block_size_ +
         position % paged_block_size_) * config_->kv_dim_;
    float* key_ptr = const_cast<float*>(
        get_buffer(ModelBufferType::kKeyCache).ptr<float>(static_cast<int64_t>(offset)));
    float* value_ptr = const_cast<float*>(
        get_buffer(ModelBufferType::kValueCache).ptr<float>(static_cast<int64_t>(offset)));
    Tensor key(DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, key_ptr);
    Tensor value(DataType::kDataTypeFp32, config_->kv_dim_, false, nullptr, value_ptr);
    key.set_device_type(device_type_);
    value.set_device_type(device_type_);
    return {key, value};
}

Status Qwen3Model::attention_mha_paged(int32_t layer, int32_t position) const
{
    const Tensor output = get_buffer(ModelBufferType::kOutputMHA);
    const Tensor& table = device_type_ == DeviceType::kDeviceCUDA
                              ? paged_block_table_device_ : paged_block_table_host_;
    if (device_type_ == DeviceType::kDeviceCPU)
        paged_attention_cpu(position, config_->head_num_, layer, paged_num_blocks_,
            paged_block_size_, config_->seq_len_, config_->kv_dim_, config_->kv_mul_,
            config_->head_size_, output, get_buffer(ModelBufferType::kQuery),
            get_buffer(ModelBufferType::kScoreStorage), get_buffer(ModelBufferType::kKeyCache),
            get_buffer(ModelBufferType::kValueCache), table);
    else
        paged_attention_cuda(position, config_->head_num_, layer, paged_num_blocks_,
            paged_block_size_, config_->seq_len_, config_->kv_dim_, config_->kv_mul_,
            config_->head_size_, output, get_buffer(ModelBufferType::kQuery),
            get_buffer(ModelBufferType::kScoreStorage), get_buffer(ModelBufferType::kKeyCache),
            get_buffer(ModelBufferType::kValueCache), table, cuda_config_.get());
    return qwen3_layers_->wo_layers_.at(layer)->forward(
        output, get_buffer(ModelBufferType::kAttnOutput));
}

Status Qwen3Model::forward_paged(const Tensor& input, const Tensor& pos_tensor) const
{
    if (paged_num_blocks_ <= 0 or paged_block_table_ids_.empty())
        return InternalError("Paged KV cache is not configured.");
    if (input.is_empty() or not config_ or input.device_type() != device_type_ or
        input.data_type() != DataType::kDataTypeFp32 or
        input.size() != static_cast<size_t>(config_->dim_))
        return InvalidArgument("Invalid Qwen3 hidden-state input.");
    if (pos_tensor.is_empty() or pos_tensor.device_type() != DeviceType::kDeviceCPU or
        pos_tensor.data_type() != DataType::kDataTypeInt32 or pos_tensor.size() != 1)
        return InvalidArgument("Invalid Qwen3 position tensor.");
    const int32_t position = pos_tensor.ptr<int32_t>()[0];
    if (position < 0 or position >= config_->seq_len_ or
        static_cast<size_t>(position / paged_block_size_) >= paged_block_table_ids_.size())
        return InvalidArgument("Paged position exceeds the configured context.");

    for (int32_t layer = 0; layer < config_->layer_num_; ++layer)
    {
        Status status = attention_rms(layer, input);
        if (not status) return status;
        Tensor query = get_buffer(ModelBufferType::kQuery);
        const auto [key, value] = slice_paged_kv_cache(layer, position);
        const Tensor normalized = get_buffer(ModelBufferType::kOutputRMSNorm);
        status = qwen3_layers_->wq_layers_.at(layer)->forward(normalized, query);
        if (not status) return status;
        status = qwen3_layers_->wk_layers_.at(layer)->forward(normalized, key);
        if (not status) return status;
        status = qwen3_layers_->wv_layers_.at(layer)->forward(normalized, value);
        if (not status) return status;
        Tensor query_heads(DataType::kDataTypeFp32, config_->head_num_, config_->head_size_,
                           false, nullptr, query.ptr<float>());
        Tensor key_heads(DataType::kDataTypeFp32, config_->kv_head_num_, config_->head_size_,
                         false, nullptr, const_cast<float*>(key.ptr<float>()));
        query_heads.set_device_type(device_type_);
        key_heads.set_device_type(device_type_);
        status = qwen3_layers_->qnorm_layers_.at(layer)->forward(query_heads, query_heads);
        if (not status) return status;
        status = qwen3_layers_->knorm_layers_.at(layer)->forward(key_heads, key_heads);
        if (not status) return status;
        if (device_type_ == DeviceType::kDeviceCPU)
            qwen3_rope_kernel_cpu(config_->head_num_, config_->kv_head_num_,
                config_->head_size_, query, key, pos_tensor,
                get_buffer(ModelBufferType::kSinCache), get_buffer(ModelBufferType::kCosCache),
                nullptr);
        else
            qwen3_rope_kernel_cu(config_->head_num_, config_->kv_head_num_,
                config_->head_size_, query, key, pos_tensor,
                get_buffer(ModelBufferType::kSinCache), get_buffer(ModelBufferType::kCosCache),
                cuda_config_->stream);
        status = attention_mha_paged(layer, position);
        if (not status) return status;
        status = feed_forward(layer, input);
        if (not status) return status;
    }
    return cls_logits(input);
}

Status Qwen3Model::forward_paged_token(int32_t token_id, int32_t position)
{
    EmbeddingOutput embedding_output(Tensor{}, Tensor{}, Tensor{});
    Status status = embedding({token_id}, embedding_output);
    if (not status) return status;
    Tensor& position_tensor = get_buffer(ModelBufferType::kInputPos);
    position_tensor.index<int32_t>(0) = position;
    return forward_paged(fill_input(position_tensor, embedding_output, false), position_tensor);
}

Status Qwen3Model::copy_logits_to_host(std::vector<float>& logits) const
{
    Tensor output = get_buffer(ModelBufferType::kForwardOutput);
    if (device_type_ == DeviceType::kDeviceCUDA) output.to_cpu();
    if (output.is_empty()) return InternalError("Qwen3 logits buffer is empty.");
    logits.assign(output.ptr<float>(), output.ptr<float>() + output.size());
    return Success();
}

std::vector<int32_t> Qwen3Model::tokenize_prompt(const std::string& prompt) const
{
    return encode(prompt);
}

std::string Qwen3Model::decode_tokens(const std::vector<int32_t>& token_ids) const
{
    return decode(token_ids);
}
}
