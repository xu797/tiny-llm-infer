#ifndef MYVLLM_UTILS_CUDA_ALLOC_H_
#define MYVLLM_UTILS_CUDA_ALLOC_H_

#include "device_alloc.h"

namespace my_vllm
{

class CUDADeviceAllocator : public DeviceAllocator
{
public:
    explicit CUDADeviceAllocator();

    void* allocate(size_t byte_size) const override;

    void release(void* ptr) const override;

private:
    //mutable声明，允许在const函数内部修改成员
    mutable std::map<int, size_t> no_busy_cnt_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> big_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> cuda_buffers_map_;
};

struct CudaMemoryBuffer 
{
    void* data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void* data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocatorFactory
{
public:
    static std::shared_ptr<CUDADeviceAllocator> get_instance() 
    {
        if (instance == nullptr) 
        {
            instance = std::make_shared<CUDADeviceAllocator>();
        }
        return instance;
    }

private:
    static std::shared_ptr<CUDADeviceAllocator> instance;
};

}
#endif