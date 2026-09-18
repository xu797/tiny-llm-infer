#ifndef MYVLLM_UTILS_CPU_ALLOC_H_
#define MYVLLM_UTILS_CPU_ALLOC_H_

#include "device_alloc.h"

namespace my_vllm
{

class CPUDeviceAllocator : public DeviceAllocator 
{
public:
    explicit CPUDeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;
};

class CPUDeviceAllocatorFactory
{
public:
    static std::shared_ptr<CPUDeviceAllocator> get_instance() 
    {
        if (instance == nullptr) 
        {
            instance = std::make_shared<CPUDeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CPUDeviceAllocator> instance;
};

}
#endif