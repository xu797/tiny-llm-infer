#include <cuda_runtime.h>

#include "cuda_alloc.h"

namespace my_vllm
{
namespace
{
constexpr size_t kIdleBufferCacheLimit = size_t{1} << 30;

void release_idle(std::vector<CudaMemoryBuffer>& buffers, int device_id)
{
    std::vector<CudaMemoryBuffer> busy_buffers;
    busy_buffers.reserve(buffers.size());
    for (const CudaMemoryBuffer& buffer : buffers)
    {
        if (buffer.busy)
        {
            busy_buffers.push_back(buffer);
            continue;
        }
        const cudaError_t state = cudaFree(buffer.data);
        CHECK(state == cudaSuccess)
            << "Error: CUDA error when releasing cached memory on device " << device_id;
    }
    buffers.swap(busy_buffers);
}
}

CUDADeviceAllocator::CUDADeviceAllocator() : DeviceAllocator(DeviceType::kDeviceCUDA)
{

}

void* CUDADeviceAllocator::allocate(size_t byte_size) const
{
    int id = -1;
    cudaError_t state = cudaGetDevice(&id); //id:gpu编号
    CHECK(state == cudaSuccess);
    //1M以上先去big_map里面去找
    if (byte_size > 1024 * 1024) 
    {
        auto& big_buffers = big_buffers_map_[id]; //单卡就是拿到GPU0的
        int sel_id = -1;
        for (int i = 0; i < big_buffers.size(); i++)
        {
            if (big_buffers[i].byte_size >= byte_size && !big_buffers[i].busy &&
                big_buffers[i].byte_size - byte_size < 1 * 1024 * 1024) 
                {
                    if (sel_id == -1 || big_buffers[sel_id].byte_size > big_buffers[i].byte_size)
                    {
                        sel_id = i; //拿到显存大于1M的最小的那一块buffer
                    }
                }
        }

        if (sel_id != -1) 
        {
            big_no_busy_cnt_[id] -= big_buffers[sel_id].byte_size;
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }
        //如果sel_id == -1, 说明缓存没命中
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
        if (state != cudaSuccess)
        {
            release_idle_buffers(id);
            state = cudaMalloc(&ptr, byte_size);
        }
        if (cudaSuccess != state) 
        {
            char buf[256];
            snprintf(buf, 256, "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory left on  device.", byte_size >> 20);
            LOG(ERROR) << buf;
            return nullptr;
        }
        big_buffers.emplace_back(ptr, byte_size, true);
        return ptr;
    }
    //小于1M就走下面这个
    auto& cuda_buffers = cuda_buffers_map_[id];
    for (int i = 0; i < cuda_buffers.size(); i++) 
    {
        if (cuda_buffers[i].byte_size >= byte_size && !cuda_buffers[i].busy) 
        {
            cuda_buffers[i].busy = true;
            no_busy_cnt_[id] -= cuda_buffers[i].byte_size;
            return cuda_buffers[i].data;
        }
    }
    void* ptr = nullptr;
    state = cudaMalloc(&ptr, byte_size);
    if (state != cudaSuccess)
    {
        release_idle_buffers(id);
        state = cudaMalloc(&ptr, byte_size);
    }
    if (cudaSuccess != state) 
    {
        char buf[256];
        snprintf(buf, 256, "Error: CUDA error when allocating %lu MB memory! maybe there's no enough memory "
                "left on  device.",
                byte_size >> 20);
        LOG(ERROR) << buf;
        return nullptr;
    }
    cuda_buffers.emplace_back(ptr, byte_size, true);
    return ptr;

}

void CUDADeviceAllocator::release(void* ptr) const
{
    if (!ptr) 
    {
        return;
    }
    for (auto& it : cuda_buffers_map_) 
    {
        for (CudaMemoryBuffer& buffer : it.second)
        {
            if (buffer.data == ptr)
            {
                if (!buffer.busy) return;
                no_busy_cnt_[it.first] += buffer.byte_size;
                buffer.busy = false;
                trim_idle_buffers_if_needed(it.first);
                return;
            }
        }
    }
    for (auto& it : big_buffers_map_)
    {
        for (CudaMemoryBuffer& buffer : it.second)
        {
            if (buffer.data == ptr)
            {
                if (!buffer.busy) return;
                big_no_busy_cnt_[it.first] += buffer.byte_size;
                buffer.busy = false;
                trim_idle_buffers_if_needed(it.first);
                return;
            }
        }
    }
    const cudaError_t state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

void CUDADeviceAllocator::release_idle_buffers(int device_id) const
{
    int current_device = device_id;
    const cudaError_t get_device_state = cudaGetDevice(&current_device);
    CHECK(get_device_state == cudaSuccess);
    if (current_device != device_id)
    {
        const cudaError_t set_device_state = cudaSetDevice(device_id);
        CHECK(set_device_state == cudaSuccess);
    }

    release_idle(cuda_buffers_map_[device_id], device_id);
    release_idle(big_buffers_map_[device_id], device_id);
    no_busy_cnt_[device_id] = 0;
    big_no_busy_cnt_[device_id] = 0;

    if (current_device != device_id)
    {
        const cudaError_t restore_state = cudaSetDevice(current_device);
        CHECK(restore_state == cudaSuccess);
    }
}

void CUDADeviceAllocator::trim_idle_buffers_if_needed(int device_id) const
{
    const size_t small_idle = no_busy_cnt_[device_id];
    const size_t big_idle = big_no_busy_cnt_[device_id];
    if (big_idle > kIdleBufferCacheLimit ||
        small_idle > kIdleBufferCacheLimit - big_idle)
    {
        release_idle_buffers(device_id);
    }
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}
