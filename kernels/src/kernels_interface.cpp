#include "kernels_interface.h"
#include "add_kernel_cpu.h"
#include "add_kernel_cuda.cuh"
#include "matmul_kernel_cpu.h"
#include "matmul_kernel_cuda.cuh"

namespace my_vllm
{
    
AddKernel get_add_kernel(DeviceType device_type) 
{
    if (device_type == DeviceType::kDeviceCPU) 
    {
        return add_kernel_cpu;
    } 
    else if (device_type == DeviceType::kDeviceCUDA) 
    {
        return add_kernel_cu;
    } 
    else {
        LOG(FATAL) << "Unknown device type for get a add kernel.";
        return nullptr;
    }
}

MatmulKernel get_matmul_kernel(DeviceType device_type) 
{
    if (device_type == DeviceType::kDeviceCPU) 
    {
        return matmul_kernel_cpu;
    } 
    else if (device_type == DeviceType::kDeviceCUDA) 
    {
        return matmul_kernel_cu;
    } 
    else 
    {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

MatmulKernelQuant get_matmul_kernel_quant8(DeviceType device_type) 
{
    if (device_type == DeviceType::kDeviceCUDA)
    {
        return matmul_kernel_cu_qint8;
    } 
    else 
    {
        LOG(FATAL) << "Unknown device type for get an matmul kernel.";
        return nullptr;
    }
}

}