#include "cuda_alloc.h"

namespace my_vllm
{
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
            big_buffers[sel_id].busy = true;
            return big_buffers[sel_id].data;
        }
        //如果sel_id == -1, 说明缓存没命中
        void* ptr = nullptr;
        state = cudaMalloc(&ptr, byte_size);
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
    if (cuda_buffers_map_.empty())
    {
        return;
    }
    // 当某张 GPU 的小块缓存空闲总容量超过 1GB，就把这张卡小块池里所有空闲 buffer 全部 cudaFree 释放，只保留正在被占用的 buffer。
    cudaError_t state = cudaSuccess;
    for (auto& it : cuda_buffers_map_) 
    {
        if (no_busy_cnt_[it.first] > 1024 * 1024 * 1024)
        {
            auto& cuda_buffers = it.second;
            std::vector<CudaMemoryBuffer> temp;
            for (int i = 0; i < cuda_buffers.size(); i++) 
            {
                if (!cuda_buffers[i].busy) 
                {
                    state = cudaSetDevice(it.first);
                    state = cudaFree(cuda_buffers[i].data);
                    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device " << it.first;
                } 
                else 
                {
                    temp.push_back(cuda_buffers[i]); //拷贝操作
                }
            }
            cuda_buffers.clear();
            it.second = temp;
            no_busy_cnt_[it.first] = 0;
        }
    }

    for (auto& it : cuda_buffers_map_) 
    {
        auto& cuda_buffers = it.second;
        for (int i = 0; i < cuda_buffers.size(); i++)
        {
            if (cuda_buffers[i].data == ptr) 
            {
                no_busy_cnt_[it.first] += cuda_buffers[i].byte_size;
                cuda_buffers[i].busy = false;
                return;
            }
        }
        auto& big_buffers = big_buffers_map_[it.first];
        for (int i = 0; i < big_buffers.size(); i++) 
        {
            if (big_buffers[i].data == ptr) 
            {
                big_buffers[i].busy = false;
                return;
            }
        }
    }
    state = cudaFree(ptr);
    CHECK(state == cudaSuccess) << "Error: CUDA error when release memory on device";
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance = nullptr;

}