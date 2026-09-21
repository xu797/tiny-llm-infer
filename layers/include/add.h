#ifndef MYVLLM_LAYERS_ADD_H_
#define MYVLLM_LAYERS_ADD_H_

#include "status.h"
#include "layer.h"

namespace my_vllm 
{

class VecAddLayer : public Layer 
{
public:
    explicit VecAddLayer(DeviceType device_type);

    Status check() const override;

    Status forward() override;
};

}  // namespace my_vllm
#endif  // MYVLLM_LAYERS_ADD_H_