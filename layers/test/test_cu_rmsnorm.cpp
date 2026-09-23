#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "kernels_interface.h"
#include "utils.cuh"
#include "buffer.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"

using namespace my_vllm;
TEST(test_rmsnorm_cu, rmsnorm_nostream)
{
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 15;

  Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor wei_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor out_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i)
  {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }
  Tensor in_cu = in_cpu.clone();
  Tensor wei_cu = wei_cpu.clone();
  Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);

  get_rmsnorm_kernel(DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu,
                                                            nullptr);
  out_cu.to_cpu();

  get_rmsnorm_kernel(DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                           nullptr);

  for (int i = 0; i < size; ++i)
  {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
}

TEST(test_rmsnorm_cu, rmsnorm_stream)
{
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32;

  Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor wei_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor out_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i)
  {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  Tensor in_cu = in_cpu.clone();
  Tensor wei_cu = wei_cpu.clone();
  Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  get_rmsnorm_kernel(DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu,
                                                            stream);
  out_cu.to_cpu();

  get_rmsnorm_kernel(DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                           nullptr);

  for (int i = 0; i < size; ++i)
  {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}

TEST(test_rmsnorm_cu, rmsnorm_stream2)
{
  auto alloc_cu = CUDADeviceAllocatorFactory::get_instance();
  auto alloc_cpu = CPUDeviceAllocatorFactory::get_instance();

  int32_t size = 32 * 151 * 15;

  Tensor in_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor wei_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);
  Tensor out_cpu(DataType::kDataTypeFp32, size, true, alloc_cpu);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_real_distribution<float> dist(0.f, 1.f);
  for (int i = 0; i < size; ++i)
  {
    in_cpu.index<float>(i) = dist(mt);
    wei_cpu.index<float>(i) = dist(mt);
  }

  Tensor in_cu = in_cpu.clone();
  Tensor wei_cu = wei_cpu.clone();
  Tensor out_cu = out_cpu.clone();
  in_cu.to_cuda(nullptr);
  wei_cu.to_cuda(nullptr);
  out_cu.to_cuda(nullptr);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  get_rmsnorm_kernel(DeviceType::kDeviceCUDA)(in_cu, wei_cu, out_cu,
                                                            stream);
  out_cu.to_cpu();

  get_rmsnorm_kernel(DeviceType::kDeviceCPU)(in_cpu, wei_cpu, out_cpu,
                                                           nullptr);

  for (int i = 0; i < size; ++i)
  {
    ASSERT_NEAR(out_cu.index<float>(i), out_cpu.index<float>(i), 1e-5f);
  }
  cudaStreamDestroy(stream);
}