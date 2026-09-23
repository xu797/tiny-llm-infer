#include <cuda_runtime_api.h>
#include <armadillo>
// #include "kernels/cpu/rmsnorm_kernel.h"

#include "kernels_interface.h"
#include "rmsnorm.h"

namespace my_vllm
{
RmsNormLayer::RmsNormLayer(DeviceType device_type, int32_t dim)
    : LayerParam(device_type, LayerType::kLayerRMSNorm, false, "RMSNorm"), dim_(dim)
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
    get_rmsnorm_kernel(device_type_)(input, weight, output, cuda_config_ ? cuda_config_->stream : nullptr);
    return Success();
}

Status RmsNormLayer::check() const
{
    // auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    // if (!status)
    // {
    //     LOG(ERROR) << "The input tensor error in the rmsnorm layer.";
    //     return status;
    // }

    auto status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim_);
    if (!status)
    {
        LOG(ERROR) << "The weight tensor error in the rmsnorm layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim_);
    if (!status)
    {
        LOG(ERROR) << "The output tensor error in the rmsnorm layer.";
        return status;
    }
    return Success();
}

}
