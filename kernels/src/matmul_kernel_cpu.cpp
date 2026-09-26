#include "matmul_kernel_cpu.h"

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

    arma::fmat input_matrix(const_cast<float*>(input.ptr<float>()),
                            input_dim, rows, false, true);
    arma::fmat weight_matrix(const_cast<float*>(weight.ptr<float>()),
                             input_dim, output_dim, false, true);
    arma::fmat output_matrix(const_cast<float*>(output.ptr<float>()),
                             output_dim, rows, false, true);
    output_matrix = (weight_matrix.t() * input_matrix) * scale;
}
}  // namespace my_vllm
