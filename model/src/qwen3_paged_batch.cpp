#include "qwen3.h"

#include <algorithm>
#include <cstring>

#include <cuda_runtime_api.h>

#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "paged_attention.h"
#include "rope_kernel_cpu.h"
#include "rope_kernel_cuda.cuh"
#include "rmsnorm.h"

namespace my_vllm
{
namespace
{
std::shared_ptr<DeviceAllocator> allocator_for(DeviceType device)
{
    if (device == DeviceType::kDeviceCPU)
        return CPUDeviceAllocatorFactory::get_instance();
    return CUDADeviceAllocatorFactory::get_instance();
}

Tensor allocated_tensor(DataType type, std::vector<int32_t> dims,
                        const std::shared_ptr<DeviceAllocator>& allocator)
{
    return Tensor(type, std::move(dims), true, allocator);
}

Tensor tensor_view(DataType type, std::vector<int32_t> dims, void* ptr,
                   DeviceType device)
{
    Tensor view(type, std::move(dims), false, nullptr, ptr);
    view.set_device_type(device);
    return view;
}
}  // namespace

Status Qwen3Model::forward_paged_batch(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const std::vector<std::vector<int32_t>>& block_tables,
    const std::vector<size_t>& sample_rows, std::vector<std::vector<float>>& logits)
{
    logits.clear();
    if (paged_num_blocks_ <= 0 || paged_block_size_ <= 0 || paged_max_batch_tokens_ <= 0)
        return InternalError("Configure the paged KV cache before running a batch.");
    if (token_ids.empty() || token_ids.size() != positions.size() ||
        token_ids.size() != block_tables.size() ||
        token_ids.size() > static_cast<size_t>(paged_max_batch_tokens_))
        return InvalidArgument("The Qwen3 batch token, position, or block table count is invalid.");

    const size_t batch_size = token_ids.size();
    const int32_t rows = static_cast<int32_t>(batch_size);
    const int32_t max_table_entries = paged_max_table_entries_;
    for (size_t row = 0; row < batch_size; ++row)
    {
        if (positions[row] < 0 || positions[row] >= config_->seq_len_)
            return InvalidArgument("A Qwen3 batch position is outside the context window.");
        const size_t logical_block =
            static_cast<size_t>(positions[row] / paged_block_size_);
        if (block_tables[row].empty() ||
            block_tables[row].size() > static_cast<size_t>(max_table_entries) ||
            logical_block >= block_tables[row].size())
            return InvalidArgument("A Qwen3 batch block table does not cover its token position.");
        for (const int32_t block : block_tables[row])
            if (block < 0 || block >= paged_num_blocks_)
                return InvalidArgument("A Qwen3 batch block table contains an invalid page id.");
    }
    for (const size_t row : sample_rows)
        if (row >= batch_size)
            return InvalidArgument("A Qwen3 batch sample row is outside the token batch.");

    const auto allocator = allocator_for(device_type_);
    EmbeddingOutput embedding_output(Tensor{}, Tensor{}, Tensor{});
    Status status = embedding(token_ids, embedding_output);
    if (!status) return status;
    Tensor hidden = embedding_output.input_embeddings;

    Tensor host_tables = allocated_tensor(
        DataType::kDataTypeInt32,
        {static_cast<int32_t>(batch_size), max_table_entries},
        CPUDeviceAllocatorFactory::get_instance());
    int32_t* host_table_data = host_tables.ptr<int32_t>();
    std::fill(host_table_data, host_table_data + host_tables.size(), -1);
    for (size_t row = 0; row < batch_size; ++row)
    {
        std::copy(block_tables[row].begin(), block_tables[row].end(),
                  host_table_data + row * static_cast<size_t>(max_table_entries));
    }

    Tensor attention_tables = host_tables;
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        attention_tables = allocated_tensor(
            DataType::kDataTypeInt32,
            {static_cast<int32_t>(batch_size), max_table_entries}, allocator);
        const size_t bytes = host_tables.byte_size();
        const cudaError_t error = cudaMemcpyAsync(
            attention_tables.ptr<int32_t>(), host_tables.ptr<int32_t>(), bytes,
            cudaMemcpyHostToDevice, cuda_config_->stream);
        if (error != cudaSuccess)
            return InternalError(std::string("Failed to upload batch block tables: ") +
                                 cudaGetErrorString(error));
    }

    Tensor host_positions = allocated_tensor(
        DataType::kDataTypeInt32, {rows}, CPUDeviceAllocatorFactory::get_instance());
    std::copy(positions.begin(), positions.end(), host_positions.ptr<int32_t>());
    Tensor attention_positions = host_positions;
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        attention_positions = allocated_tensor(DataType::kDataTypeInt32, {rows}, allocator);
        const cudaError_t error = cudaMemcpyAsync(
            attention_positions.ptr<int32_t>(), host_positions.ptr<int32_t>(),
            host_positions.byte_size(), cudaMemcpyHostToDevice, cuda_config_->stream);
        if (error != cudaSuccess)
            return InternalError(std::string("Failed to upload batch positions: ") +
                                 cudaGetErrorString(error));
    }

    const int32_t dim = config_->dim_;
    const int32_t query_dim = config_->query_dim_;
    const int32_t kv_dim = config_->kv_dim_;
    const int32_t hidden_dim = config_->hidden_dim_;
    Tensor position_tensor(DataType::kDataTypeInt32, 1, true,
                           CPUDeviceAllocatorFactory::get_instance());
    const std::vector<int32_t> score_dims =
        device_type_ == DeviceType::kDeviceCUDA
            ? std::vector<int32_t>{rows, config_->head_num_, config_->seq_len_}
            : std::vector<int32_t>{config_->head_num_, config_->seq_len_};
    Tensor attention_scores = allocated_tensor(DataType::kDataTypeFp32, score_dims, allocator);
    Tensor pos_view(DataType::kDataTypeInt32, 1, false, nullptr,
                    position_tensor.ptr<int32_t>());
    pos_view.set_device_type(DeviceType::kDeviceCPU);

    for (int32_t layer = 0; layer < config_->layer_num_; ++layer)
    {
        Tensor normalized = allocated_tensor(DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).input_norm_->forward(hidden, normalized);
        if (!status) return status;

        Tensor query = allocated_tensor(DataType::kDataTypeFp32, {rows, query_dim}, allocator);
        Tensor keys = allocated_tensor(DataType::kDataTypeFp32, {rows, kv_dim}, allocator);
        Tensor values = allocated_tensor(DataType::kDataTypeFp32, {rows, kv_dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).q_proj_->forward(normalized, query);
        if (!status) return status;
        status = qwen3_layers_->decoder_layers_.at(layer).k_proj_->forward(normalized, keys);
        if (!status) return status;
        status = qwen3_layers_->decoder_layers_.at(layer).v_proj_->forward(normalized, values);
        if (!status) return status;

        Tensor query_heads = tensor_view(
            DataType::kDataTypeFp32,
            {rows * config_->head_num_, config_->head_size_}, query.ptr<float>(),
            device_type_);
        Tensor key_heads = tensor_view(
            DataType::kDataTypeFp32,
            {rows * config_->kv_head_num_, config_->head_size_}, keys.ptr<float>(),
            device_type_);
        status = qwen3_layers_->decoder_layers_.at(layer).q_norm_->forward(query_heads, query_heads);
        if (!status) return status;
        status = qwen3_layers_->decoder_layers_.at(layer).k_norm_->forward(key_heads, key_heads);
        if (!status) return status;

        if (device_type_ == DeviceType::kDeviceCPU)
        {
            float* key_cache = const_cast<float*>(
                get_buffer(ModelBufferType::kKeyCache).ptr<float>());
            float* value_cache = const_cast<float*>(
                get_buffer(ModelBufferType::kValueCache).ptr<float>());
            for (int32_t row = 0; row < rows; ++row)
            {
                pos_view.index<int32_t>(0) = positions[static_cast<size_t>(row)];
                Tensor query_row = tensor_view(
                    DataType::kDataTypeFp32, {query_dim},
                    query.ptr<float>(static_cast<int64_t>(row) * query_dim), device_type_);
                Tensor key_row = tensor_view(
                    DataType::kDataTypeFp32, {kv_dim},
                    keys.ptr<float>(static_cast<int64_t>(row) * kv_dim), device_type_);
                qwen3_rope_kernel_cpu(
                    config_->head_num_, config_->kv_head_num_, config_->head_size_,
                    query_row, key_row, pos_view, get_buffer(ModelBufferType::kSinCache),
                    get_buffer(ModelBufferType::kCosCache), nullptr);

                const size_t position = static_cast<size_t>(positions[static_cast<size_t>(row)]);
                const size_t page = position / static_cast<size_t>(paged_block_size_);
                const size_t offset =
                    ((static_cast<size_t>(layer) * static_cast<size_t>(paged_num_blocks_) +
                      static_cast<size_t>(block_tables[static_cast<size_t>(row)][page])) *
                         static_cast<size_t>(paged_block_size_) +
                     position % static_cast<size_t>(paged_block_size_)) *
                    static_cast<size_t>(kv_dim);
                const float* key_source =
                    keys.ptr<float>(static_cast<int64_t>(row) * kv_dim);
                const float* value_source =
                    values.ptr<float>(static_cast<int64_t>(row) * kv_dim);
                std::memcpy(key_cache + offset, key_source,
                            static_cast<size_t>(kv_dim) * sizeof(float));
                std::memcpy(value_cache + offset, value_source,
                            static_cast<size_t>(kv_dim) * sizeof(float));
            }
        }
        else
        {
            qwen3_rope_batch_kernel_cu(
                rows, config_->head_num_, config_->kv_head_num_, config_->head_size_,
                query, keys, attention_positions, get_buffer(ModelBufferType::kSinCache),
                get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
            paged_kv_cache_store_batch_cuda(
                rows, layer, paged_num_blocks_, paged_block_size_, kv_dim,
                max_table_entries, attention_positions, attention_tables, keys, values,
                get_buffer(ModelBufferType::kKeyCache),
                get_buffer(ModelBufferType::kValueCache), cuda_config_.get());
        }

        Tensor mha_output = allocated_tensor(
            DataType::kDataTypeFp32, {rows, query_dim}, allocator);
        if (device_type_ == DeviceType::kDeviceCPU)
        {
            for (int32_t row = 0; row < rows; ++row)
            {
                const int32_t position = positions[static_cast<size_t>(row)];
                Tensor query_row = tensor_view(
                    DataType::kDataTypeFp32, {query_dim},
                    query.ptr<float>(static_cast<int64_t>(row) * query_dim), device_type_);
                Tensor output_row = tensor_view(
                    DataType::kDataTypeFp32, {query_dim},
                    mha_output.ptr<float>(static_cast<int64_t>(row) * query_dim), device_type_);
                const int32_t* table_data = attention_tables.ptr<int32_t>() +
                    static_cast<size_t>(row) * static_cast<size_t>(max_table_entries);
                Tensor table_row = tensor_view(
                    DataType::kDataTypeInt32, {max_table_entries},
                    const_cast<int32_t*>(table_data), device_type_);
                paged_attention_cpu(
                    position, config_->head_num_, layer, paged_num_blocks_,
                    paged_block_size_, config_->seq_len_, kv_dim, config_->kv_mul_,
                    config_->head_size_, output_row, query_row, attention_scores,
                    get_buffer(ModelBufferType::kKeyCache),
                    get_buffer(ModelBufferType::kValueCache), table_row);
            }
        }
        else
        {
            paged_attention_batch_cuda(
                rows, config_->head_num_, layer, paged_num_blocks_, paged_block_size_,
                config_->seq_len_, max_table_entries, kv_dim, config_->kv_mul_,
                config_->head_size_, attention_positions, mha_output, query,
                attention_scores, get_buffer(ModelBufferType::kKeyCache),
                get_buffer(ModelBufferType::kValueCache), attention_tables,
                cuda_config_.get());
        }

        Tensor attention_output = allocated_tensor(
            DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).o_proj_->forward(mha_output,
                                                               attention_output);
        if (!status) return status;

        Tensor residual = allocated_tensor(DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->add_layer_->forward(hidden, attention_output, residual);
        if (!status) return status;
        Tensor ffn_normalized = allocated_tensor(
            DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).post_attention_norm_->forward(residual, ffn_normalized);
        if (!status) return status;

        Tensor gate = allocated_tensor(
            DataType::kDataTypeFp32, {rows, hidden_dim}, allocator);
        Tensor up = allocated_tensor(
            DataType::kDataTypeFp32, {rows, hidden_dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).gate_proj_->forward(ffn_normalized, gate);
        if (!status) return status;
        status = qwen3_layers_->decoder_layers_.at(layer).up_proj_->forward(ffn_normalized, up);
        if (!status) return status;
        Tensor activated = allocated_tensor(
            DataType::kDataTypeFp32, {rows, hidden_dim}, allocator);
        status = qwen3_layers_->swiglu_layer_->forward(gate, up, activated);
        if (!status) return status;
        Tensor down = allocated_tensor(DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->decoder_layers_.at(layer).down_proj_->forward(activated, down);
        if (!status) return status;

        Tensor next_hidden = allocated_tensor(
            DataType::kDataTypeFp32, {rows, dim}, allocator);
        status = qwen3_layers_->add_layer_->forward(residual, down, next_hidden);
        if (!status) return status;
        hidden = std::move(next_hidden);
    }

    Tensor final_hidden = allocated_tensor(
        DataType::kDataTypeFp32, {rows, dim}, allocator);
    status = qwen3_layers_->final_norm_->forward(hidden, final_hidden);
    if (!status) return status;

    logits.reserve(sample_rows.size());
    for (const size_t row : sample_rows)
    {
        Tensor hidden_row = tensor_view(
            DataType::kDataTypeFp32, {dim},
            final_hidden.ptr<float>(static_cast<int64_t>(row) * dim), device_type_);
        Tensor row_logits = allocated_tensor(
            DataType::kDataTypeFp32, {config_->vocab_size_}, allocator);
        status = qwen3_layers_->lm_head_->forward(hidden_row, row_logits);
        if (!status) return status;
        if (device_type_ == DeviceType::kDeviceCUDA) row_logits.to_cpu();
        logits.emplace_back(row_logits.ptr<float>(),
                            row_logits.ptr<float>() + row_logits.size());
    }
    return Success();
}
}  // namespace my_vllm
