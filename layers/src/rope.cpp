#include <cmath>

// #include "kernels/cpu/rope_kernel.h"
#include "kernels_interface.h"
#include "rope.h"

namespace my_vllm 
{
RoPELayer::RoPELayer(DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size)
    : Layer(device_type, LayerType::kLayerRoPe, "RoPe"),
      dim_(dim),
      kv_dim_(kv_dim),
      head_size_(head_size) 
{
    reset_input_size(5);
    reset_output_size(1);
}

Status RoPELayer::forward() 
{
    Status status = check();
    if (!status) {
        return status;
    }

    Tensor input_q = this->get_input(0);
    Tensor input_k = this->get_input(1);
    Tensor input_pos = this->get_input(2);

    Tensor sin_cache = this->get_input(3);
    Tensor cos_cache = this->get_input(4);

    if (device_type_ == DeviceType::kDeviceCUDA) 
    {
        CHECK(cuda_config_ != nullptr);
    }
    get_rope_kernel(device_type_)(dim_, kv_dim_, head_size_, input_q, input_k, input_pos,
                                            sin_cache, cos_cache,
                                            cuda_config_ ? cuda_config_->stream : nullptr);
    return Success();
}

Status RoPELayer::check() const 
{
    // pos tensor
    auto status = check_tensor_with_dim(get_input(2), DeviceType::kDeviceCPU,
                                        DataType::kDataTypeInt32, 1);
    if (!status) 
    {
        LOG(ERROR) << "The input tensor 2 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(get_input(1), device_type_, data_type_, kv_dim_);
    if (!status) 
    {
        LOG(ERROR) << "The input tensor 1 error in the add layer.";
        return status;
    }

    status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim_);
    if (!status) 
    {
        LOG(ERROR) << "The input tensor 0 error in the add layer.";
        return status;
    }
    return Success();
    }

}  // namespace op