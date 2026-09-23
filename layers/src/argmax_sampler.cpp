#include <algorithm>

#include "argmax_kernel_cuda.cuh"
#include "argmax_sampler.h"

namespace my_vllm 
{
size_t ArgmaxSampler::sample(const float* logits, size_t size, void* stream) 
{
    if (device_type_ == DeviceType::kDeviceCPU)
    {
        size_t next = std::distance(logits, std::max_element(logits, logits + size));
        return next;
    } 
    else 
    {
        size_t next = argmax_kernel_cu(logits, size, stream);
        return next;
    }
}
} 