#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "kernels_interface.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "utils.cuh"
#include "buffer.h"


TEST(test_matmul_cu, matmul_linear_stream5)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    Tensor input(DataType::kDataTypeFp32, 4, true, alloc_cpu);
    Tensor weight(DataType::kDataTypeFp32, 4, 4, true, alloc_cpu);

    for (int i = 0; i < 4; ++i)
    {
        input.index<float>(i) = float(i);
    }

    for (int i = 0; i < 16; ++i)
    {
        weight.index<float>(i) = float(i);
    }
    Tensor input_cpu = input.clone();
    Tensor weight_cpu = weight.clone();

    input.to_cuda(nullptr);
    weight.to_cuda(nullptr);

    Tensor out_cu(DataType::kDataTypeFp32, 4, true, alloc_cu);
    Tensor out_cpu(DataType::kDataTypeFp32, 4, true, alloc_cpu);

    CudaConfig *config = new CudaConfig;
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    config->stream = stream;
    get_matmul_kernel(DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, config);

    get_matmul_kernel(DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f, config);

    out_cu.to_cpu();
    for (int i = 0; i < out_cu.size(); ++i)
    {
        ASSERT_EQ(out_cu.index<float>(i), out_cpu.index<float>(i));
    }
}

TEST(test_matmul_cu, matmul_linear_course)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    Tensor input(DataType::kDataTypeFp32, 3, true, alloc_cpu);
    Tensor weight(DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

    input.index<float>(0) = float(1);
    input.index<float>(1) = float(1);
    input.index<float>(2) = float(-1);

    for (int i = 1; i <= 9; ++i)
    {
      weight.index<float>(i - 1) = float(i);
    }
    Tensor input_cpu = input.clone();
    Tensor weight_cpu = weight.clone();

    input.to_cuda(nullptr);
    weight.to_cuda(nullptr);

    Tensor out_cpu(DataType::kDataTypeFp32, 3, true, alloc_cpu);

    get_matmul_kernel(DeviceType::kDeviceCPU)(input_cpu, weight_cpu, out_cpu, 1.f,
                                                            nullptr);

    ASSERT_EQ(out_cpu.index<float>(0), 0);
    ASSERT_EQ(out_cpu.index<float>(1), 3);
    ASSERT_EQ(out_cpu.index<float>(2), 6);
}

TEST(test_matmul_cu, matmul_linear_course_cuda)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    Tensor input(DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);
    Tensor weight(DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

    input.index<float>(0) = float(1);
    input.index<float>(1) = float(1);
    input.index<float>(2) = float(-1);
    for (int i = 1; i <= 9; ++i)
    {
        weight.index<float>(i - 1) = float(i);
    }

    input.to_cuda();
    weight.to_cuda();

    Tensor out_cu(DataType::kDataTypeFp32, 3, true, alloc_cu);

    get_matmul_kernel(DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

    Tensor out_cpu = out_cu.clone();
    out_cpu.to_cpu();

    ASSERT_EQ(out_cpu.index<float>(0), 0);
    ASSERT_EQ(out_cpu.index<float>(1), 3);
    ASSERT_EQ(out_cpu.index<float>(2), 6);
}

// TEST(test_matmul_cu, matmul_linear_2d_two_col)
// {
//     using namespace my_vllm;
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

//     // input shape [3,2]：in_dim0=3, in_dim1=2，满足 in_dim0 == wei_dim1=3
//     Tensor input(DataType::kDataTypeFp32, 3, 2, true, alloc_cpu);
//     Tensor weight(DataType::kDataTypeFp32, 3, 3, true, alloc_cpu);

//     // 第0列向量 [1,1,-1]
//     input.index<float>(0) = 1.0f;
//     input.index<float>(1) = 1.0f;
//     input.index<float>(2) = -1.0f;
//     // 第1列向量 [1,2,3]
//     input.index<float>(3) = 1.0f;
//     input.index<float>(4) = 2.0f;
//     input.index<float>(5) = 3.0f;

//     for (int i = 1; i <= 9; ++i)
//     {
//         weight.index<float>(i - 1) = float(i);
//     }

//     Tensor input_cpu = input.clone();
//     Tensor weight_cpu = weight.clone();
//     input.to_cuda(nullptr);
//     weight.to_cuda(nullptr);

//     // output shape [3, 2]
//     Tensor out_cu(DataType::kDataTypeFp32, 3, 2, true, alloc_cu);
//     get_matmul_kernel(DeviceType::kDeviceCUDA)(input, weight, out_cu, 1.f, nullptr);

//     printf("out[0][0] = %f\n", out_cu.index<float>(0));
//     printf("out[1][0] = %f\n", out_cu.index<float>(1));
//     printf("out[2][0] = %f\n", out_cu.index<float>(2));

//     printf("out[0][1] = %f\n", out_cu.index<float>(3));
//     printf("out[1][1] = %f\n", out_cu.index<float>(4));
//     printf("out[2][1] = %f\n", out_cu.index<float>(5));
// }

