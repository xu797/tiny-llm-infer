#ifndef MYVLLM_LAYERS_RMSNORM_H_
#define MYVLLM_LAYERS_RMSNORM_H_

#include "layer.h"

namespace my_vllm 
{
class RmsNormLayer : public LayerParam 
{
public:
    explicit RmsNormLayer(DeviceType device_type, int32_t dim);

    Status check() const override;

    Status forward() override;

private:
    int32_t dim_ = 0;
};
}  
#endif  // MYVLLM_LAYERS_RMSNORM_H_
