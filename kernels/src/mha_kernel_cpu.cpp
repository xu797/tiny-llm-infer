#include <cuda_runtime_api.h>

#include "mha_kernel_cpu.h"
#include "kernels_interface.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"

namespace my_vllm 
{
void mha_kernel(int32_t pos, int32_t head_num, int32_t layer_index, int32_t seq_len, int32_t kv_dim,
                int32_t kv_mul, int32_t head_size, const Tensor& mha_out,
                const Tensor& query_tensor, const Tensor& score_tensor,
                const Tensor& key_cache_tensor, const Tensor& value_cache_tensor,
                DeviceType device_type, CudaConfig* config) 
{
    int32_t layer_offset = layer_index * seq_len * kv_dim;
    float scale = 1.f / std::sqrt(static_cast<float>(head_size)); //head_size:每个头的向量维度

    std::shared_ptr<DeviceAllocator> allocator;
    if (device_type == DeviceType::kDeviceCPU) 
    {
        allocator = CPUDeviceAllocatorFactory::get_instance();
    } 
    else 
    {
        allocator = CUDADeviceAllocatorFactory::get_instance();
    }
    for (int32_t h = 0; h < head_num; ++h) 
    {
        float* score_head_addr = const_cast<float*>(score_tensor.ptr<float>() + h * seq_len);
        float* query_head_addr = const_cast<float*>(query_tensor.ptr<float>() + h * head_size);


        Tensor query_mat(DataType::kDataTypeFp32, head_size, false, nullptr, query_head_addr);
        query_mat.set_device_type(device_type);

        for (int32_t t = 0; t <= pos; t++) 
        {
            int32_t cache_offset = t * kv_dim + (h / kv_mul) * head_size;
            const float* key_head_addr = key_cache_tensor.ptr<float>() + layer_offset + cache_offset;
            Tensor key_mat(DataType::kDataTypeFp32, 1, head_size, false, nullptr, const_cast<float*>(key_head_addr));
            Tensor score_mat(DataType::kDataTypeFp32, 1, false, nullptr, score_head_addr + t);
            key_mat.set_device_type(device_type);
            score_mat.set_device_type(device_type);
            get_matmul_kernel(device_type)(query_mat, key_mat, score_mat, scale, config);
        }

        Tensor score_head_tensor(DataType::kDataTypeFp32, pos + 1, false, nullptr, score_head_addr);
        score_head_tensor.set_device_type(device_type);
        get_softmax_kernel(device_type)(score_head_tensor, config ? config->stream : nullptr);

        float* output_head_ptr = const_cast<float*>(mha_out.ptr<float>()) + h * head_size;
        allocator->memset_zero(output_head_ptr, sizeof(float) * head_size, config ? config->stream : nullptr, false);
        Tensor output_tensor(DataType::kDataTypeFp32, head_size, false, nullptr, output_head_ptr);
        output_tensor.set_device_type(device_type);

        int32_t cache_offset = (h / kv_mul) * head_size;
        float* value_head_addr = const_cast<float*>(value_cache_tensor.ptr<float>()) + layer_offset + cache_offset;
        Tensor value_tensor(DataType::kDataTypeFp32, head_size, false, nullptr, value_head_addr);
        get_scale_sum_kernel(device_type)(value_tensor, score_head_tensor, output_tensor, pos, head_size, kv_dim, config ? config->stream : nullptr);
    }
}
}  