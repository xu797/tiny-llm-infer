#ifndef MYVLLM_LAYERS_ARGMAX_SAMPLE_H_
#define MYVLLM_LAYERS_ARGMAX_SAMPLE_H_

#include "status.h"
#include "sampler.h"

namespace my_vllm 
{
class ArgmaxSampler : public Sampler 
{
public:
    explicit ArgmaxSampler(DeviceType device_type) : Sampler(device_type) {}

    size_t sample(const float* logits, size_t size, void* stream) override;
};
} 
#endif  // MYVLLM_LAYERS_ARGMAX_SAMPLE_H_
