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
    Status status;
    const int32_t input_tensor_num = 2;
    for (int32_t i = 0; i < input_tensor_num; ++i) 
    {
        status = check_tensor_with_dim(get_input(0), device_type_, data_type_, hidden_dim_);
        if (!status) 
        {
            LOG(ERROR) << "The input tensor " << std::to_string(i) << " error in the swiglu layer.";
            return status;
        }
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, hidden_dim_);
    if (!status) 
    {
        LOG(ERROR) << "The output tensor error in the swiglu layer.";
        return status;
    }
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
