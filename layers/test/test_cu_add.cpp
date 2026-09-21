#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "kernels_interface.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"
#include "utils.cuh"
#include "buffer.h"
#include "tensor.h"

TEST(test_add_cu, add1_nostream)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor t2(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor out(DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(static_cast<float *>(t1.get_buffer()->ptr()), size, 2.f);
    set_value_cu(static_cast<float *>(t2.get_buffer()->ptr()), size, 3.f);

    get_add_kernel(DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
    cudaDeviceSynchronize();
    float *output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(output[i], 5.f);
    }

    delete[] output;
}

TEST(test_add_cu, add1_stream)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151;

    Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor t2(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor out(DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(static_cast<float *>(t1.get_buffer()->ptr()), size, 2.f);
    set_value_cu(static_cast<float *>(t2.get_buffer()->ptr()), size, 3.f);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    get_add_kernel(DeviceType::kDeviceCUDA)(t1, t2, out, stream);
    cudaDeviceSynchronize();
    float *output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(output[i], 5.f);
    }
    cudaStreamDestroy(stream);
    delete[] output;
}

TEST(test_add_cu, add_align1)
{
    using namespace my_vllm;
    auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();

    int32_t size = 32 * 151 * 13;

    Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor t2(DataType::kDataTypeFp32, size, true, alloc_cu);
    Tensor out(DataType::kDataTypeFp32, size, true, alloc_cu);

    set_value_cu(static_cast<float *>(t1.get_buffer()->ptr()), size, 2.1f);
    set_value_cu(static_cast<float *>(t2.get_buffer()->ptr()), size, 3.3f);

    get_add_kernel(DeviceType::kDeviceCUDA)(t1, t2, out, nullptr);
    cudaDeviceSynchronize();
    float *output = new float[size];
    cudaMemcpy(output, out.ptr<float>(), size * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(output[i], 5.4f, 0.1f);
    }

    delete[] output;
}