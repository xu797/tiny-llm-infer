#ifndef MYVLLM_LAYERS_MATMUL_H_
#define MYVLLM_LAYERS_MATMUL_H_

#include "layer.h"
#include "cuda_config.h"

namespace my_vllm 
{
class MatmulLayer : public LayerParam 
{
public:
    explicit MatmulLayer(DeviceType device_type, int32_t dim0, int32_t dim1);

    Status check() const override;

    Status forward() override;

private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
};
}  // namespace my_vllm
#endif  // MYVLLM_LAYERS_MATMUL_H_
