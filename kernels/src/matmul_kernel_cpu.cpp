#include "matmul_kernel_cpu.h"
#include "fp16.h"

namespace my_vllm
{
void matmul_kernel_cpu(const Tensor& input, const Tensor& weight,
                       const Tensor& output, float scale, const CudaConfig* config)
{
    UNUSED(config);
    CHECK(!input.is_empty() && !weight.is_empty() && !output.is_empty());
    CHECK(input.device_type() == DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == DeviceType::kDeviceCPU);
    CHECK(output.device_type() == DeviceType::kDeviceCPU);
    CHECK(input.dims_size() == 1 || input.dims_size() == 2);
    CHECK_EQ(weight.dims_size(), 2);

    const int32_t rows = input.dims_size() == 1 ? 1 : input.get_dim(0);
    const int32_t input_dim = input.get_dim(input.dims_size() - 1);
    const int32_t output_dim = weight.get_dim(0);
    CHECK_EQ(weight.get_dim(1), input_dim);
    CHECK_EQ(output.size(), static_cast<size_t>(rows) * output_dim);

    const uint16_t* x = input.ptr<uint16_t>();
    const uint16_t* w = weight.ptr<uint16_t>();
    uint16_t* y = const_cast<uint16_t*>(output.ptr<uint16_t>());
    #pragma omp parallel for collapse(2)
    for (int32_t row = 0; row < rows; ++row)
        for (int32_t out = 0; out < output_dim; ++out)
        {
            float sum = 0.0f;
            for (int32_t col = 0; col < input_dim; ++col)
                sum += fp16_to_float(x[static_cast<size_t>(row) * input_dim + col]) *
                       fp16_to_float(w[static_cast<size_t>(out) * input_dim + col]);
            y[static_cast<size_t>(row) * output_dim + out] = float_to_fp16(sum * scale);
        }
}

void matmul_kernel_cpu_int8(const Tensor& input, const Tensor& weight,
                            const Tensor& scales, const Tensor& output,
                            int32_t group_size)
{
    const int32_t rows = input.dims_size() == 1 ? 1 : input.get_dim(0);
    const int32_t input_dim = input.get_dim(input.dims_size() - 1);
    const int32_t output_dim = weight.get_dim(0);
    const int32_t groups_per_row = (input_dim + group_size - 1) / group_size;
    const int8_t* quantized = weight.ptr<int8_t>();
    const float* scales_ptr = scales.ptr<float>();
    uint16_t* result = const_cast<uint16_t*>(output.ptr<uint16_t>());
    #pragma omp parallel for collapse(2)
    for (int32_t row = 0; row < rows; ++row)
    {
        for (int32_t out = 0; out < output_dim; ++out)
        {
            float sum = 0.0f;
            const int8_t* weight_row = quantized + static_cast<size_t>(out) * input_dim;
            const float* row_scales = scales_ptr + static_cast<size_t>(out) * groups_per_row;
            for (int32_t col = 0; col < input_dim; ++col)
            sum += fp16_to_float(input.ptr<uint16_t>()[static_cast<size_t>(row) * input_dim + col]) *
                       static_cast<float>(weight_row[col]) * row_scales[col / group_size];
            result[static_cast<size_t>(row) * output_dim + out] = float_to_fp16(sum);
        }
    }
}
}  // namespace my_vllm
