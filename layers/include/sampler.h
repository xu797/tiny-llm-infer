#ifndef MYVLLM_LAYERS_SAMPLE_H_
#define MYVLLM_LAYERS_SAMPLE_H_

#include <cstddef>
#include <cstdint>

namespace my_vllm 
{
class Sampler {
public:
    explicit Sampler(DeviceType device_type) : device_type_(device_type) {}

    virtual size_t sample(const float* logits, size_t size, void* stream = nullptr) = 0;

protected:
    DeviceType device_type_;
};
} 
#endif  
