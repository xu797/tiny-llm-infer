#include "matmul.h"
#include "matmul_kernel_cpu.h"
#include "matmul_kernel_cuda.cuh"
#include "kernels_interface.h"

namespace my_vllm
{
MatmulLayer::MatmulLayer(DeviceType device_type, int32_t dim0, int32_t dim1)
    : LayerParam(device_type, LayerType::kLayerMatmul, "Matmul"),
      dim0_(dim0), dim1_(dim1)
{
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
}

  Status MatmulLayer::check() const
  {
    const Tensor& input = get_input(0);
    const Tensor& weight = get_weight(0);
    const Tensor& output = get_output(0);
    if (input.is_empty() || input.device_type() != device_type_ ||
        input.data_type() != data_type_ ||
        (input.dims_size() != 1 && input.dims_size() != 2) ||
        input.get_dim(input.dims_size() - 1) != dim1_)
      return InvalidArgument("The input tensor shape is invalid in the matmul layer.");

    if (weight.is_empty() || weight.device_type() != device_type_ ||
        weight.data_type() != data_type_ || weight.dims_size() != 2 ||
        weight.get_dim(0) != dim0_ || weight.get_dim(1) != dim1_)
      return InvalidArgument("The weight tensor shape is invalid in the matmul layer.");

    const bool output_shape_valid =
        input.dims_size() == 1
            ? output.dims_size() == 1 && output.get_dim(0) == dim0_
            : output.dims_size() == 2 && output.get_dim(0) == input.get_dim(0) &&
                  output.get_dim(1) == dim0_;
    if (output.is_empty() || output.device_type() != device_type_ ||
        output.data_type() != data_type_ || !output_shape_valid)
      return InvalidArgument("The output tensor shape is invalid in the matmul layer.");
    return Success();
  }

  Status MatmulLayer::forward()
  {
    auto status = check();
    if (!status)
    {
      return status;
    }
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
      CHECK(cuda_config_ != nullptr);
    }
    get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                    cuda_config_ ? cuda_config_.get() : nullptr);

    return Success();
  }


}  // namespace my_vllm
