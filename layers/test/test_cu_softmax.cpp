// #include <cuda_runtime_api.h>
// #include <glog/logging.h>
// #include <gtest/gtest.h>

// #include "kernels_interface.h"
// #include "utils.cuh"
// #include "buffer.h"
// #include "cpu_alloc.h"
// #include "cuda_alloc.h"

// using namespace my_vllm;
// TEST(test_softmax_cu, softmax_nostream)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

//     int32_t size = 32 * 151;

//     Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

//     srand(0);
//     for (int i = 0; i < size; ++i)
//     {
//         in_cpu.index<float>(i) = rand() % 31;
//     }

//     Tensor in_cu = in_cpu.clone();
//     in_cu.to_cuda();

//     get_softmax_kernel(DeviceType::kDeviceCUDA)(in_cu, nullptr);
//     get_softmax_kernel(DeviceType::kDeviceCPU)(in_cpu, nullptr);

//     in_cu.to_cpu();

//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
//     }
// }

// TEST(test_softmax_cu, softmax_stream)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

//     int32_t size = 72 * 151;

//     Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

//     srand(0);
//     for (int i = 0; i < size; ++i)
//     {
//         in_cpu.index<float>(i) = rand() % 31;
//     }

//     Tensor in_cu = in_cpu.clone();
//     in_cu.to_cuda();

//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     get_softmax_kernel(DeviceType::kDeviceCUDA)(in_cu, stream);
//     get_softmax_kernel(DeviceType::kDeviceCPU)(in_cpu, nullptr);

//     in_cu.to_cpu();

//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
//     }
// }

// TEST(test_softmax_cu, softmax_stream2)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

//     int32_t size = 72 * 18;

//     Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_real_distribution<float> dist(0.f, 1.f);
//     for (int i = 0; i < size; ++i)
//     {
//         in_cpu.index<float>(i) = dist(mt);
//     }

//     Tensor in_cu = in_cpu.clone();
//     in_cu.to_cuda();

//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     get_softmax_kernel(DeviceType::kDeviceCUDA)(in_cu, stream);
//     get_softmax_kernel(DeviceType::kDeviceCPU)(in_cpu, nullptr);
//     in_cu.to_cpu();

//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
//     }
// }

// TEST(test_softmax_cu, softmax_stream3)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

//     int32_t size = 1;

//     Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_real_distribution<float> dist(0.f, 1.f);
//     for (int i = 0; i < size; ++i)
//     {
//         in_cpu.index<float>(i) = dist(mt);
//     }

//     Tensor in_cu = in_cpu.clone();
//     in_cu.to_cuda();

//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     get_softmax_kernel(DeviceType::kDeviceCUDA)(in_cu, stream);
//     get_softmax_kernel(DeviceType::kDeviceCPU)(in_cpu, nullptr);
//     in_cu.to_cpu();

//     for (int i = 0; i < size; ++i)
//     {
//         ASSERT_NEAR(in_cpu.index<float>(i), in_cu.index<float>(i), 1e-5f);
//     }
// }