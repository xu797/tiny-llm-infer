#include "matmul.h"
#include "matmul_kernel_cpu.h"
#include "matmul_kernel_cuda.cuh"
#include "kernels_interface.h"

namespace my_vllm
{
  MatmulLayer::MatmulLayer(DeviceType device_type, int32_t dim0, int32_t dim1,
                           bool is_quant_layer, bool has_bias)
      : LayerParam(device_type, LayerType::kLayerMatmul, is_quant_layer, "Matmul"),
        dim0_(dim0),
        dim1_(dim1),
        has_bias_(has_bias)
  {
    reset_input_size(1);
    reset_output_size(1);
    reset_weight_size(1);
    if (has_bias_)
    {
      bias_.resize(1);
    }
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

    const DataType weight_type = is_quant_layer_ ? DataType::kDataTypeInt8 : data_type_;
    if (weight.is_empty() || weight.device_type() != device_type_ ||
        weight.data_type() != weight_type || weight.dims_size() != 2 ||
        weight.get_dim(0) != dim0_ || weight.get_dim(1) != dim1_)
      return InvalidArgument("The weight tensor shape is invalid in the matmul layer.");

    if (is_quant_layer_)
    {
      const Status status = check_tensor(scales_, device_type_, DataType::kDataTypeFp32);
      if (!status) return status;
    }

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
    if (is_quant_layer_)
    {
      get_matmul_kernel_quant8(device_type_)(get_input(0), get_weight(0), get_output(0),
                                             group_size_, scales_,
                                             cuda_config_ ? cuda_config_.get() : nullptr);
    }
    else
    {
      get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                      cuda_config_ ? cuda_config_.get() : nullptr);
    }

    if (has_bias_)
    {
      get_add_kernel(device_type_)(get_output(0), get_bias(0), get_output(0),
                                   cuda_config_ ? cuda_config_->stream : nullptr);
    }

    return Success();
  }

  Status MatmulLayer::set_bias(int32_t idx, int32_t &dim, const void *bias_ptr,
                               DeviceType device_type)
  {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    CHECK_NE(bias_ptr, nullptr);

    size_t size = dim * sizeof(float);
    std::shared_ptr<Buffer> buffer =
        std::make_shared<Buffer>(size, nullptr, const_cast<void *>(bias_ptr), true);
    if (device_type != DeviceType::kDeviceUnknown)
    {
      buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_)
    {
      Tensor bias(DataType::kDataTypeFp32, dim);
      bias.set_device_type(device_type);
      CHECK(bias.assign(buffer));
      // LOG(INFO) << "bias:" << bias.index<float>(0);
      bias_.at(idx) = bias;
    }
    else
    {
      // is quant layer
      Tensor bias(DataType::kDataTypeInt8, dim);
      bias.set_device_type(device_type);
      CHECK(bias.assign(buffer));
      bias_.at(idx) = bias;

      const int32_t bias_size = static_cast<int32_t>(bias.size());
      CHECK(bias_size % group_size_ == 0);

      int32_t scale_nums = bias_size / group_size_;
      scales_ = Tensor{DataType::kDataTypeFp32, scale_nums, false, nullptr,
                       reinterpret_cast<float *>((int8_t *)bias_ptr + bias_size)};
      scales_.set_device_type(device_type);
    }

    return Success();
  }

  Tensor &MatmulLayer::get_bias(int32_t idx)
  {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
  }

  const Tensor &MatmulLayer::get_bias(int32_t idx) const
  {
    CHECK_GE(idx, 0);
    CHECK_LT(idx, bias_.size());
    return bias_.at(idx);
  }

  void MatmulLayer::to_cuda()
  {
    LayerParam::to_cuda();
    if (has_bias_)
    {
      for (auto &bias : bias_)
      {
        bias.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
      }
    }
  }

}