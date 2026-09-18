#ifndef MYVVLM_UTILS_ALLOC_FACTORY_H_
#define MYVVLM_UTILS_ALLOC_FACTORY_H_

#include <memory>
#include "device_alloc.h"

namespace my_vllm
{
class DeviceAllocatorFactory 
{
public:
    static std::shared_ptr<DeviceAllocator> get_instance(DeviceType device_type);
};
}
#endif
