#ifndef MYVLLM_LAYERS_MATMUL_H_
#define MYVLLM_LAYERS_MATMUL_H_

#include "layer.h"
#include "cuda_config.h"

namespace my_vllm 
{
class MatmulLayer : public LayerParam 
{
public:
    explicit MatmulLayer(DeviceType device_type, int32_t dim0, int32_t dim1,
                        bool is_quant_layer = false, bool has_bias = false);

    Status check() const override;

    Status forward() override;

    Status set_bias(int32_t idx, int32_t& dims, const void* bias_ptr,
                            DeviceType device_type);

    Tensor& get_bias(int32_t idx);

    const Tensor& get_bias(int32_t idx) const;

    void to_cuda() override;

private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
    bool has_bias_ = false;
    std::vector<Tensor> bias_;
};
}  // namespace my_vllm
#endif  // MYVLLM_LAYERS_MATMUL_H_
