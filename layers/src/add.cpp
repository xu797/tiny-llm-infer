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
    const Tensor& input1 = get_input(0);
    const Tensor& input2 = get_input(1);
    const Tensor& output = get_output(0);
    if (input1.is_empty() || input2.is_empty() || output.is_empty() ||
        input1.device_type() != device_type_ || input2.device_type() != device_type_ ||
        output.device_type() != device_type_ || input1.data_type() != data_type_ ||
        input2.data_type() != data_type_ || output.data_type() != data_type_ ||
        input1.dims() != input2.dims() || input1.dims() != output.dims())
        return InvalidArgument("The add layer tensors must have the same shape and device.");
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