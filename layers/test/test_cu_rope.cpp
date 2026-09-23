// #include <cuda_runtime_api.h>
// #include <glog/logging.h>
// #include <gtest/gtest.h>

// #include "kernels_interface.h"
// #include "utils.cuh"
// #include "buffer.h"
// #include "cpu_alloc.h"
// #include "cuda_alloc.h"

// using namespace my_vllm;
// TEST(test_rope_cu, rope_nostream)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
//     int32_t dim = 256;
//     int32_t head_size = 64;
//     int32_t kv_dim = 128;
//     int32_t pos = 3;
//     Tensor input_pos(DataType::kDataTypeInt32, 1, true, alloc_cpu);
//     input_pos.index<int32_t>(0) = pos;

//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_real_distribution<float> dist(0.f, 1.f);
//     Tensor input_q_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);
//     Tensor input_k_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);

//     for (int i = 0; i < dim; ++i)
//     {
//         input_q_cpu.index<float>(i) = dist(mt);
//         input_k_cpu.index<float>(i) = dist(mt);
//     }

//     Tensor input_q_gpu = input_q_cpu.clone();
//     Tensor input_k_gpu = input_k_cpu.clone();
//     input_q_gpu.to_cuda(nullptr);
//     input_k_gpu.to_cuda(nullptr);

//     get_rope_kernel(DeviceType::kDeviceCPU)(
//         dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos, nullptr);

//     get_rope_kernel(DeviceType::kDeviceCUDA)(
//         dim, kv_dim, head_size, input_q_gpu, input_k_gpu, input_pos, nullptr);
//     cudaDeviceSynchronize();

//     input_q_gpu.to_cpu();
//     input_k_gpu.to_cpu();
//     for (int32_t i = 0; i < dim; ++i)
//     {
//         ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f)
//             << "ik: " << i;
//         ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f)
//             << "iq: " << i;
//     }
// }

// TEST(test_rope_cu, rope_nostream2)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
//     int32_t dim = 512;
//     int32_t head_size = 128;
//     int32_t kv_dim = 32;
//     int32_t pos = 4;
//     Tensor input_pos(DataType::kDataTypeInt32, 1, true, alloc_cpu);
//     input_pos.index<int32_t>(0) = pos;

//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_real_distribution<float> dist(0.f, 1.f);
//     Tensor input_q_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);
//     Tensor input_k_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);

//     for (int i = 0; i < dim; ++i)
//     {
//         input_q_cpu.index<float>(i) = dist(mt);
//         input_k_cpu.index<float>(i) = dist(mt);
//     }

//     Tensor input_q_gpu = input_q_cpu.clone();
//     Tensor input_k_gpu = input_k_cpu.clone();
//     input_q_gpu.to_cuda(nullptr);
//     input_k_gpu.to_cuda(nullptr);

//     get_rope_kernel(DeviceType::kDeviceCPU)(
//         dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos, nullptr);

//     get_rope_kernel(DeviceType::kDeviceCUDA)(
//         dim, kv_dim, head_size, input_q_gpu, input_k_gpu, input_pos, nullptr);
//     cudaDeviceSynchronize();

//     input_q_gpu.to_cpu();
//     input_k_gpu.to_cpu();
//     for (int32_t i = 0; i < dim; ++i)
//     {
//         ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f)
//             << "ik: " << i;
//         ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f)
//             << "iq: " << i;
//     }
// }

// TEST(test_rope_cu, rope_stream1)
// {
//     auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
//     auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
//     int32_t dim = 512;
//     int32_t head_size = 128;
//     int32_t kv_dim = 32;
//     int32_t pos = 4;
//     Tensor input_pos(DataType::kDataTypeInt32, 1, true, alloc_cpu);
//     input_pos.index<int32_t>(0) = pos;

//     std::random_device rd;
//     std::mt19937 mt(rd());
//     std::uniform_real_distribution<float> dist(0.f, 1.f);
//     Tensor input_q_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);
//     Tensor input_k_cpu(DataType::kDataTypeFp32, dim, true, alloc_cpu);
//     cudaStream_t stream;
//     cudaStreamCreate(&stream);
//     for (int i = 0; i < dim; ++i)
//     {
//         input_q_cpu.index<float>(i) = dist(mt);
//         input_k_cpu.index<float>(i) = dist(mt);
//     }

//     Tensor input_q_gpu = input_q_cpu.clone();
//     Tensor input_k_gpu = input_k_cpu.clone();
//     input_q_gpu.to_cuda(nullptr);
//     input_k_gpu.to_cuda(nullptr);

//     get_rope_kernel(DeviceType::kDeviceCPU)(
//         dim, kv_dim, head_size, input_q_cpu, input_k_cpu, input_pos, nullptr);

//     get_rope_kernel(DeviceType::kDeviceCUDA)(
//         dim, kv_dim, head_size, input_q_gpu, input_k_gpu, input_pos, stream);
//     cudaDeviceSynchronize();

//     input_q_gpu.to_cpu();
//     input_k_gpu.to_cpu();
//     for (int32_t i = 0; i < dim; ++i)
//     {
//         ASSERT_NEAR(input_k_cpu.index<float>(i), input_k_gpu.index<float>(i), 1e-3f)
//             << "ik: " << i;
//         ASSERT_NEAR(input_q_cpu.index<float>(i), input_q_gpu.index<float>(i), 1e-3f)
//             << "iq: " << i;
//     }
// }