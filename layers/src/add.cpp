#include "add.h"
#include "kernels_interface.h"

namespace my_vllm 
{

VecAddLayer::VecAddLayer(DeviceType device_type)
    : Layer(device_type, LayerType::kLayerAdd, "Add") 
{
    reset_input_size(2);
    reset_output_size(1);
}

Status VecAddLayer::check() const 
{
    Tensor input1 = this->get_input(0);
    Tensor input2 = this->get_input(1);
    int32_t size = input1.size();
    Status status;
    status = check_tensor_with_dim(input1, device_type_, data_type_, size);
    if (!status) 
    {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(input2, device_type_, data_type_, size);
    if (!status)
    {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, size);
    if (!status) 
    {
        LOG(ERROR) << "The output tensor error in the add layer.";
        return status;
    }
    return Success();
}

Status VecAddLayer::forward()
{
    auto status = this->check();
    if (!status) 
    {
        return status;
    }
    auto input1 = this->get_input(0);
    auto input2 = this->get_input(1);
    auto output = this->get_output(0);
    if (device_type_ == DeviceType::kDeviceCUDA) 
    {
        CHECK(cuda_config_ != nullptr);
    }
    get_add_kernel(device_type_)(input1, input2, output, cuda_config_ ? cuda_config_->stream : nullptr);
    return Success();
}

}  // namespace my_vllm