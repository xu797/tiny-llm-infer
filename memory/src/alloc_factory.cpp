#include <glog/logging.h>

#include "alloc_factory.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"

namespace my_vllm
{
std::shared_ptr<DeviceAllocator> DeviceAllocatorFactory::get_instance(DeviceType device_type)
{
    if (device_type == DeviceType::kDeviceCPU) 
    {
        return CPUDeviceAllocatorFactory::get_instance();
    } 
    else if (device_type == DeviceType::kDeviceCUDA) 
    {
        return CUDADeviceAllocatorFactory::get_instance();
    } 
    else 
    {
        LOG(FATAL) << "This device type of allocator is not supported!";
        return nullptr;
    }
}
}
