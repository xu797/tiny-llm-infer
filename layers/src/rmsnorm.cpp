#include <cuda_runtime_api.h>
#include <armadillo>
// #include "kernels/cpu/rmsnorm_kernel.h"

#include "kernels_interface.h"
#include "rmsnorm.h"

namespace my_vllm
{
RmsNormLayer::RmsNormLayer(DeviceType device_type, int32_t dim, float eps)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, "RMSNorm"), dim_(dim), eps_(eps)
{
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

Status RmsNormLayer::forward()
{
    auto status = check();
    if (!status)
    {
        return status;
    }
    auto input = this->get_input(0);
    auto weight = this->get_weight(0);
    auto output = this->get_output(0);
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        CHECK(cuda_config_ != nullptr);
    }
    get_rmsnorm_kernel(device_type_)(input, weight, output,
                                     cuda_config_ ? cuda_config_->stream : nullptr, eps_);
    return Success();
}

Status RmsNormLayer::check() const
{
    const Tensor& input = get_input(0);
    const Tensor& weight = get_weight(0);
    const Tensor& output = get_output(0);
    if (input.is_empty() || input.data_type() != data_type_ || input.device_type() != device_type_)
    {
        return InvalidArgument("The input tensor is invalid in the rmsnorm layer.");
    }
    if (input.dims_size() < 1 || input.get_dim(input.dims_size() - 1) != dim_)
    {
        return InvalidArgument("The input tensor's last dimension must match the rmsnorm width.");
    }
    if (weight.is_empty() || weight.data_type() != data_type_ ||
        weight.device_type() != device_type_ || weight.size() != static_cast<size_t>(dim_))
    {
        return InvalidArgument("The weight tensor is invalid in the rmsnorm layer.");
    }
    if (output.is_empty() || output.data_type() != data_type_ ||
        output.device_type() != device_type_ || output.dims() != input.dims())
    {
        return InvalidArgument("The output tensor must match the rmsnorm input shape.");
    }
    return Success();
}

}
