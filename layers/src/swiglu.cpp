#include "swiglu.h"
// #include "swiglu_kernel.h"
#include "kernels_interface.h"
#include "layer.h"

namespace my_vllm
{
SwiGLULayer::SwiGLULayer(DeviceType device_type, int32_t hidden_dim)
    : Layer(device_type, LayerType::kLayerSwiGLU, "SwiGLU"), hidden_dim_(hidden_dim) 
{
    reset_input_size(2);
    reset_output_size(1);
}

Status SwiGLULayer::check() const
{
    const Tensor& gate = get_input(0);
    const Tensor& up = get_input(1);
    const Tensor& output = get_output(0);
    if (gate.is_empty() || up.is_empty() || output.is_empty() ||
        gate.device_type() != device_type_ || up.device_type() != device_type_ ||
        output.device_type() != device_type_ || gate.data_type() != data_type_ ||
        up.data_type() != data_type_ || output.data_type() != data_type_ ||
        gate.dims() != up.dims() || gate.dims() != output.dims() ||
        gate.dims_size() == 0 || gate.get_dim(gate.dims_size() - 1) != hidden_dim_)
        return InvalidArgument("The SwiGLU tensors must share a shape ending in hidden_dim.");
    return Success();
}

Status SwiGLULayer::forward() 
{
    auto status = check();
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
    get_swiglu_kernel(device_type_)(input1, input2, output, cuda_config_ ? cuda_config_->stream : nullptr);
    return Success();
}

}  // namespace op
