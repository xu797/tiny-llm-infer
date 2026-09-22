#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "kernels_interface.h"
// #include "embedding.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "utils.cuh"
#include "buffer.h"

using namespace my_vllm;

TEST(test_emb_cu, emb1_nostream)
{
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;
    int32_t dim = 512;
    int32_t size = 2048;

    Tensor input(DataType::kDataTypeFp32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 1;

    Tensor weight(DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    Tensor output(DataType::kDataTypeFp32, dim, true, alloc_cu);

    for (int i = 0; i < size; ++i)
    {
        weight.index<float>(i) = static_cast<float>(i);
    }
    weight.to_cuda();
    get_emb_kernel(DeviceType::kDeviceCUDA)(input, weight, output, token, nullptr);

    output.to_cpu();
    for (int i = 0; i < dim; ++i)
    {
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }
}

TEST(test_emb_cu, emb2_nostream)
{
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;
    int32_t dim = 512;
    int32_t size = 2048;

    Tensor input(DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 2;

    Tensor weight(DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    Tensor output(DataType::kDataTypeFp32, dim, true, alloc_cu);

    for (int i = 0; i < size; ++i)
    {
        weight.index<float>(i) = static_cast<float>(i);
    }
    weight.to_cuda();
    get_emb_kernel(DeviceType::kDeviceCUDA)(input, weight, output, token, nullptr);

    output.to_cpu();
    for (int i = 0; i < dim; ++i)
    {
        ASSERT_EQ(output.index<float>(i), 1024 + i);
    }
}

TEST(test_emb_cu, emb1_stream)
{
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
    auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

    int32_t token = 4;
    int32_t dim = 512;
    int32_t size = 2048;

    Tensor input(DataType::kDataTypeInt32, 1, true, alloc_cpu);
    input.index<int32_t>(0) = 1;

    Tensor weight(DataType::kDataTypeFp32, token, dim, true, alloc_cpu);
    Tensor output(DataType::kDataTypeFp32, dim, true, alloc_cu);

    for (int i = 0; i < size; ++i)
    {
        weight.index<float>(i) = static_cast<float>(i);
    }
    weight.to_cuda();
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    get_emb_kernel(DeviceType::kDeviceCUDA)(input, weight, output, token, stream);

    output.to_cpu();
    for (int i = 0; i < dim; ++i)
    {
        ASSERT_EQ(output.index<float>(i), 512 + i);
    }

    cudaStreamDestroy(stream);
}
