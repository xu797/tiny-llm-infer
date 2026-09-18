#include "cpu_alloc.h"

namespace my_vllm
{
CPUDeviceAllocator::CPUDeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCPU)
{

}

void* CPUDeviceAllocator::allocate(size_t byte_size) const
{
    if (!byte_size)
    {
        return nullptr;
    }

    #ifdef KUIPER_HAVE_POSIX_MEMALIGN
    //windows会走这个分支
        void* data = nullptr;
        const size_t alignment = (byte_size >= size_t(1024)) ? size_t(32) : size_t(16);
        int status = posix_memalign((void**)&data,
                                    ((alignment >= sizeof(void*)) ? alignment : sizeof(void*)),
                                    byte_size);
        if (status != 0) {
            return nullptr;
        }
        return data;
    #else
        void* data = malloc(byte_size);
        return data;
        /*
        接收的时候强制类型转换即可
        void* raw_ptr = malloc(1024);
        // 转成 float 指针：把这块内存当成 float 数组看待
        float* f_ptr = (float*)raw_ptr;
        */
    #endif
}

void CPUDeviceAllocator::release(void* ptr) const 
{
    if (ptr)
    {
        free(ptr);
    }
}

std::shared_ptr<CPUDeviceAllocator> CPUDeviceAllocatorFactory::instance = nullptr;

}