#include "rmsnorm_kernel_cpu.h"
#include "fp16.h"

namespace my_vllm
{
void rmsnorm_kernel_cpu(const Tensor& input, const Tensor& weight, const Tensor& output,
                        void* stream, float eps)
{
    UNUSED(stream);
    CHECK(!input.is_empty());
    CHECK(!weight.is_empty());
    CHECK(!output.is_empty());

    CHECK(input.device_type() == DeviceType::kDeviceCPU &&
        weight.device_type() == DeviceType::kDeviceCPU &&
        output.device_type() == DeviceType::kDeviceCPU);

    const uint16_t* in_ptr = input.ptr<uint16_t>();
    const uint16_t* wei_ptr = weight.ptr<uint16_t>();
    uint16_t* out_ptr = const_cast<uint16_t*>(output.ptr<uint16_t>());
    const int32_t dim = static_cast<int32_t>(weight.size());
    CHECK_GT(dim, 0);
    CHECK_EQ(input.size() % dim, 0);
    CHECK_EQ(output.size(), input.size());
    const int32_t row_num = static_cast<int32_t>(input.size() / dim);

    for (int32_t row = 0; row < row_num; ++row)
    {
        const uint16_t* row_input = in_ptr + row * dim;
        uint16_t* row_output = out_ptr + row * dim;
        float square_sum = 0.f;
        for (int32_t col = 0; col < dim; ++col)
        {
            const float x = fp16_to_float(row_input[col]);
            square_sum += x * x;
        }
        const float scale = 1.f / std::sqrt(square_sum / dim + eps);
        for (int32_t col = 0; col < dim; ++col)
        {
            row_output[col] = float_to_fp16(fp16_to_float(wei_ptr[col]) *
                                             fp16_to_float(row_input[col]) * scale);
        }
    }
}
}
