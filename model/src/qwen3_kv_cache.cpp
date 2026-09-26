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
    if (paged_num_blocks_ > 0)
        return InvalidArgument("The paged KV cache pool has already been configured.");
    std::shared_ptr<DeviceAllocator> allocator;
    if (device_type_ == DeviceType::kDeviceCPU)
        allocator = CPUDeviceAllocatorFactory::get_instance();
    else if (device_type_ == DeviceType::kDeviceCUDA)
        allocator = CUDADeviceAllocatorFactory::get_instance();
    else
        return InvalidArgument("Initialize Qwen3 before configuring its KV cache.");

    Tensor key_cache(DataType::kDataTypeFp32, config_->layer_num_, num_blocks,
                     block_size, config_->kv_dim_, true, allocator);
    Tensor value_cache(DataType::kDataTypeFp32, config_->layer_num_, num_blocks,
                       block_size, config_->kv_dim_, true, allocator);
    if (key_cache.is_empty() || value_cache.is_empty())
        return InternalError("Failed to allocate the requested paged KV cache pool.");

    buffers_[ModelBufferType::kKeyCache] = std::move(key_cache);
    buffers_[ModelBufferType::kValueCache] = std::move(value_cache);
    paged_num_blocks_ = num_blocks;
    paged_block_size_ = block_size;
    paged_max_batch_tokens_ = max_batch_tokens;
    paged_max_table_entries_ = (config_->seq_len_ + block_size - 1) / block_size;
    return Success();
}

}  // namespace my_vllm
