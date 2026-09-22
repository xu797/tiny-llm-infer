#include "matmul_kernel_cpu.h"
#include <iostream>
namespace my_vllm
{
void matmul_kernel_cpu(const Tensor &input, const Tensor &weight,
                        const Tensor &output, float scale, const CudaConfig *config)
{
    UNUSED(config);
    CHECK(input.is_empty() == false);
    CHECK(weight.is_empty() == false);
    CHECK(output.is_empty() == false);
    CHECK(input.device_type() == DeviceType::kDeviceCPU);
    CHECK(weight.device_type() == DeviceType::kDeviceCPU);
    CHECK(output.device_type() == DeviceType::kDeviceCPU);

    const float *input_ptr = input.ptr<float>();
    const float *weight_ptr = weight.ptr<float>();
    const float *output_ptr = output.ptr<float>();

    int32_t in_dim1 = 1;
    int32_t in_dim0 = 1;
    if (input.dims_size() == 2)
    {
        in_dim0 = input.get_dim(0);
        in_dim1 = input.get_dim(1);
    }
    else if (input.dims_size() == 1)
    {
        in_dim0 = input.get_dim(0);
    }
    else
    {
        LOG(FATAL) << "The input tensor has a wrong dim size.";
    }

    CHECK_EQ(weight.dims_size(), 2);
    const int32_t wei_dim0 = weight.get_dim(0);
    const int32_t wei_dim1 = weight.get_dim(1);
    CHECK_EQ(in_dim0, wei_dim1);

    CHECK_EQ(output.size(), wei_dim0 * in_dim1);
    
    arma::fmat input_mat(const_cast<float *>(input_ptr), in_dim1, in_dim0, false, true);
    arma::fmat weight_mat(const_cast<float *>(weight_ptr), wei_dim1, wei_dim0, false, true);
    arma::fmat output_mat(const_cast<float *>(output_ptr), in_dim1, wei_dim0, false, true);
    output_mat = ((input_mat * weight_mat)) * scale;

    // std::cout << "===== input_mat =====" << std::endl;
    // std::cout << input_mat << std::endl;
    // std::cout << "===== weight_mat =====" << std::endl;
    // std::cout << weight_mat << std::endl;

    // output_mat = ((input_mat * weight_mat)) * scale;

    // std::cout << "===== output_mat after mul =====" << std::endl;
    // std::cout << output_mat << std::endl;
    }

}