// #include <cuda_runtime_api.h>
// #include <glog/logging.h>
// #include <gtest/gtest.h>

// #include "kernels_interface.h"
// #include "utils.cuh"
// #include "buffer.h"
// #include "cpu_alloc.h"
// #include "cuda_alloc.h"

// using namespace my_vllm;
// TEST(test_scale_cu, scale1_nostream)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     int32_t size = 32 * 151;

//     Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
//     set_value_cu(static_cast<float *>(t1.get_buffer()->ptr()), size, 2.f);
//     get_scale_kernel(DeviceType::kDeviceCUDA)(0.5f, t1, nullptr);
//     cudaDeviceSynchronize();

//     t1.to_cpu();
//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_EQ(t1.index<float>(i), 1.f);
//     }
// }

// TEST(test_scale_cu, scale1_stream)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     int32_t size = 32 * 151;

//     Tensor t1(DataType::kDataTypeFp32, size, true, alloc_cu);
//     set_value_cu(static_cast<float *>(t1.get_buffer()->ptr()), size, 2.f);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     get_scale_kernel(DeviceType::kDeviceCUDA)(0.4f, t1, nullptr);
//     cudaDeviceSynchronize();

//     t1.to_cpu();
//     cudaStreamDestroy(stream);

//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_EQ(t1.index<float>(i), 0.8f);
//     }
// }
