#include "matmul.h"
#include "matmul_kernel_cpu.h"
#include "matmul_kernel_cuda.cuh"
#include "kernels_interface.h"
#include "cpu_alloc.h"
#include "fp16.h"
#include <algorithm>
#include <cmath>

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

Status MatmulLayer::set_bias(const uint16_t* bias, size_t count)
{
    if (!bias || count != static_cast<size_t>(dim0_))
        return InvalidArgument("The linear bias must match the output dimension.");
    bias_ = Tensor(DataType::kDataTypeFp16, dim0_, true,
                   CPUDeviceAllocatorFactory::get_instance());
    std::copy(bias, bias + count, bias_.ptr<uint16_t>());
    return Success();
}

Status MatmulLayer::set_quantized_weight(const float* weights, size_t count,
                                         int32_t group_size)
{
    if (!weights || count != static_cast<size_t>(dim0_) * dim1_ || group_size <= 0)
        return InvalidArgument("Invalid source weights or group size for INT8 quantization.");
    const int32_t groups_per_row = (dim1_ + group_size - 1) / group_size;
    Tensor quantized(DataType::kDataTypeInt8, dim0_, dim1_, true,
                     CPUDeviceAllocatorFactory::get_instance());
    quant_scales_ = Tensor(DataType::kDataTypeFp32, dim0_ * groups_per_row, true,
                           CPUDeviceAllocatorFactory::get_instance());
    int8_t* quantized_ptr = quantized.ptr<int8_t>();
    float* scales_ptr = quant_scales_.ptr<float>();
    for (int32_t row = 0; row < dim0_; ++row)
    {
        for (int32_t group = 0; group < groups_per_row; ++group)
        {
            const int32_t begin = group * group_size;
            const int32_t end = std::min(begin + group_size, dim1_);
            float max_abs = 0.0f;
            for (int32_t col = begin; col < end; ++col)
                max_abs = std::max(max_abs, std::abs(weights[static_cast<size_t>(row) * dim1_ + col]));
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scales_ptr[static_cast<size_t>(row) * groups_per_row + group] = scale;
            for (int32_t col = begin; col < end; ++col)
            {
                const float value = weights[static_cast<size_t>(row) * dim1_ + col] / scale;
                const int32_t rounded = static_cast<int32_t>(std::nearbyint(value));
                quantized_ptr[static_cast<size_t>(row) * dim1_ + col] =
                    static_cast<int8_t>(std::max(-127, std::min(127, rounded)));
            }
        }
    }
    Status status = LayerParam::set_weight(0, quantized);
    if (!status) return status;
    quant_group_size_ = group_size;
    return Success();
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

    const DataType expected_weight_type = quant_group_size_ > 0
        ? DataType::kDataTypeInt8 : data_type_;
    if (weight.is_empty() || weight.device_type() != device_type_ ||
        weight.data_type() != expected_weight_type || weight.dims_size() != 2 ||
        weight.get_dim(0) != dim0_ || weight.get_dim(1) != dim1_)
      return InvalidArgument("The weight tensor shape is invalid in the matmul layer.");
    if (quant_group_size_ > 0 &&
        (quant_scales_.is_empty() || quant_scales_.device_type() != device_type_ ||
         quant_scales_.data_type() != DataType::kDataTypeFp32 ||
         quant_scales_.size() != static_cast<size_t>(dim0_) *
             static_cast<size_t>((dim1_ + quant_group_size_ - 1) / quant_group_size_)))
        return InvalidArgument("The quantization scales are invalid in the matmul layer.");

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
    if (quant_group_size_ > 0)
    {
        if (device_type_ == DeviceType::kDeviceCUDA)
            matmul_kernel_cu_int8(get_input(0), get_weight(0), quant_scales_, get_output(0),
                                  quant_group_size_, cuda_config_.get());
        else
            matmul_kernel_cpu_int8(get_input(0), get_weight(0), quant_scales_, get_output(0),
                                   quant_group_size_);
    }
    else
        get_matmul_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), 1.f,
                                        cuda_config_ ? cuda_config_.get() : nullptr);

    if (!bias_.is_empty())
    {
        const int32_t rows = get_input(0).dims_size() == 1 ? 1 : get_input(0).get_dim(0);
        if (device_type_ == DeviceType::kDeviceCUDA)
            add_bias_rows_cuda(get_output(0), bias_, rows, dim0_, cuda_config_.get());
        else
        {
            uint16_t* output = const_cast<uint16_t*>(get_output(0).ptr<uint16_t>());
            const uint16_t* bias = bias_.ptr<uint16_t>();
            for (int32_t row = 0; row < rows; ++row)
                for (int32_t col = 0; col < dim0_; ++col)
                    output[static_cast<size_t>(row) * dim0_ + col] = float_to_fp16(
                        fp16_to_float(output[static_cast<size_t>(row) * dim0_ + col]) +
                        fp16_to_float(bias[col]));
        }
    }

    return Success();
  }

void MatmulLayer::to_cuda()
{
    LayerParam::to_cuda();
    if (quant_group_size_ > 0) quant_scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    if (!bias_.is_empty()) bias_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
}


}  // namespace my_vllm
