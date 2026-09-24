#include "embed_kernel_cpu.h"
#include "cpu_alloc.h"

namespace my_vllm 
{

void emb_kernel_normal(const Tensor& input, const Tensor& weight,
                       const Tensor& output, int32_t vocab_size, void* stream) 
{
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    const int32_t input_num = static_cast<int32_t>(input.size()); //seq_len
    const int32_t weight_dim = weight.get_dim(1); //hidden_dim
    CHECK(weight.device_type() == output.device_type());
    CHECK(input.device_type() == DeviceType::kDeviceCPU);

    /*
    input:[seq_len]
    weight:[vocab_size, hid_dim]
    output:[seq_len, hid_dim]
    */
    const auto allocator = CPUDeviceAllocatorFactory::get_instance();
    for (int32_t i = 0; i < input_num; ++i)
    {
        int32_t token = *input.ptr<int32_t>(i);
        if (token < 0 || token >= vocab_size)
        {
            LOG(FATAL) << "Token index is outside the embedding vocabulary.";
        } 
        else 
        {
            float* dest_ptr = const_cast<float*>(output.ptr<float>(i * weight_dim));
            float* src_ptr = const_cast<float*>(weight.ptr<float>(token * weight_dim));
            if (weight.device_type() == DeviceType::kDeviceCPU) 
            {
                allocator->memcpy(src_ptr, dest_ptr, weight_dim * sizeof(float), MemcpyKind::kMemcpyCPU2CPU);
            } 
            else 
            {
                LOG(FATAL) << "Unknown device type of weight tensor in the embedding layer.";
            }
        }
    }
}

}
