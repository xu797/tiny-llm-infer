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
    auto status = check_tensor_with_dim(get_input(0), device_type_, data_type_, dim1_);
    if (!status)
    {
      LOG(ERROR) << "The input tensor error in the matmul layer.";
      return status;
    }

    if (!is_quant_layer_)
    {
      status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, dim0_, dim1_);
      if (!status)
      {
        LOG(ERROR) << "The weight tensor error in the matmul layer.";
        return status;
      }
    }
    else
    {
      status = check_tensor_with_dim(get_weight(0), device_type_, DataType::kDataTypeInt8,
                                     dim0_, dim1_);
      if (!status)
      {
        LOG(ERROR) << "The weight tensor error in the matmul layer.";
        return status;
      }
    }

    if (is_quant_layer_)
    {
      status = check_tensor_with_dim(scales_, device_type_, DataType::kDataTypeFp32, scales_.size());
      if (!status)
      {
        LOG(ERROR) << "The scale tensor error in the matmul layer.";
        return status;
      }
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, dim0_);
    if (!status)
    {
      LOG(ERROR) << "The output tensor error in the matmul layer.";
      return status;
    }
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