#include "rmsnorm_kernel_cpu.h"

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

    const float* in_ptr = input.ptr<float>();
    const float* wei_ptr = weight.ptr<float>();
    const float* out_ptr = output.ptr<float>();
    const int32_t dim = static_cast<int32_t>(weight.size());
    CHECK_GT(dim, 0);
    CHECK_EQ(input.size() % dim, 0);
    CHECK_EQ(output.size(), input.size());
    const int32_t row_num = static_cast<int32_t>(input.size() / dim);

    for (int32_t row = 0; row < row_num; ++row)
    {
        const float* row_input = in_ptr + row * dim;
        float* row_output = const_cast<float*>(out_ptr) + row * dim;
        float square_sum = 0.f;
        for (int32_t col = 0; col < dim; ++col)
        {
            square_sum += row_input[col] * row_input[col];
        }
        const float scale = 1.f / std::sqrt(square_sum / dim + eps);
        for (int32_t col = 0; col < dim; ++col)
        {
            row_output[col] = wei_ptr[col] * row_input[col] * scale;
        }
    }
}
}
