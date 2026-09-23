#ifndef MYVLLM_LAYERS_SWIGLU_H_
#define MYVLLM_LAYERS_SWIGLU_H_

#include "layer.h"

namespace my_vllm 
{
class SwiGLULayer : public Layer 
{
public:
    explicit SwiGLULayer(DeviceType device_type, int32_t hidden_dim);

    Status check() const override;

    Status forward() override;

private:
    int32_t hidden_dim_ = 0;
};
}  
#endif  // MYVLLM_LAYERS_SWIGLU_H_
