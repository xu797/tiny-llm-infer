#ifndef MYVLLM_LAYERS_MATMUL_H_
#define MYVLLM_LAYERS_MATMUL_H_

#include "layer.h"
#include "cuda_config.h"
#include <vector>

namespace my_vllm 
{
class MatmulLayer : public LayerParam 
{
public:
    explicit MatmulLayer(DeviceType device_type, int32_t dim0, int32_t dim1);
    Status set_quantized_weight(const float* weights, size_t count, int32_t group_size = 128);
    Status set_bias(const uint16_t* bias, size_t count);

    Status check() const override;

    Status forward() override;
    void to_cuda() override;

private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
    int32_t quant_group_size_ = 0;
    Tensor quant_scales_;
    Tensor bias_;
};
}  // namespace my_vllm
#endif  // MYVLLM_LAYERS_MATMUL_H_
