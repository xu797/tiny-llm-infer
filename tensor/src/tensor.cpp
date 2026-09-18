#include "tensor/tensor.h"

namespace my_vllm
{

template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init)
{
    if (begin >= end) 
    {
        return 0;
    }
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
}

Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc = false,
            std::shared_ptr<DeviceAllocator> alloc = nullptr, void *ptr = nullptr)
            : data_type_(data_type)
{
    dims_.push_back(dim0);
    size_ = dim0;
    if (need_alloc && alloc) 
    {
        allocate(alloc);
    } 
    else 
    {
        if (ptr != nullptr) 
        {
            CHECK(need_alloc == false) << "The need_alloc is is true when ptr parameter is not a null pointer.";
            init_buffer(alloc, data_type_, need_alloc, ptr);
        }
    }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void *ptr = nullptr)
                : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = dim0 * dim1;
    if (need_alloc && alloc)
    {
        allocate(alloc);
    }
    else 
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2,
                bool need_alloc = false, std::shared_ptr<DeviceAllocator> alloc = nullptr,
                void *ptr = nullptr) : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = dim0 * dim1 * dim2;
    if (need_alloc && alloc) 
    {
        allocate(alloc);
    } 
    else 
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3,
                bool need_alloc = false, std::shared_ptr<DeviceAllocator> alloc = nullptr,
                void *ptr = nullptr) : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = dim0 * dim1 * dim2 * dim3;
    if (need_alloc && alloc)
    {
        allocate(alloc);
    } 
    else
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(DataType data_type, std::vector<int32_t> dims, bool need_alloc = false,
                std::shared_ptr<DeviceAllocator> alloc = nullptr, void *ptr = nullptr)
                : dims_(std::move(dims)), data_type_(data_type)
{
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
    if (need_alloc && alloc) 
    {
        allocate(alloc);
    } 
    else 
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

bool Tensor::allocate(std::shared_ptr<DeviceAllocator> allocator, bool need_realloc) 
{
    if (!allocator) 
    {
        LOG(ERROR) << "The allocator parameter in the allocate function is null pointer!";
        return false;
    }

    size_t byte_size = this->byte_size();
    if (!byte_size)
    {
        LOG(ERROR) << "The byte_size parameter in the allocate function is equal to zero!";
        return false;
    }

    if (buffer_ && byte_size <= buffer_->byte_size())
    {
        if (!need_realloc)
        {
            return true;
        }
    }

    buffer_ = std::make_shared<Buffer>(byte_size, allocator, nullptr);
    if (!buffer_->ptr())
    {
        LOG(ERROR) << "The memory allocated is a null pointer!";
        return false;
    }
    return true;
}

void Tensor::to_cpu()
{
    CHECK_NE(buffer_, nullptr);
    const DeviceType device_type = this->device_type();

    if (device_type == DeviceType::kDeviceUnknown) 
    {
        LOG(ERROR) << "The device type of the tensor is unknown.";
    } 
    else if (device_type == DeviceType::kDeviceCUDA) 
    {
        size_t byte_size = this->byte_size();
        auto cpu_alloc = CPUDeviceAllocatorFactory::get_instance();
        auto cpu_buffer = std::make_shared<Buffer>(byte_size, cpu_alloc);
        cpu_alloc->memcpy(buffer_->ptr(), cpu_buffer->ptr(), byte_size, MemcpyKind::kMemcpyCUDA2CPU);
        this->buffer_ = cpu_buffer;
    } 
    else {
        LOG(INFO) << "The device type of the tensor is already cuda.";
    }
}

void Tensor::to_cuda(cudaStream_t stream)
{
    CHECK_NE(buffer_, nullptr);
    const DeviceType device_type = this->device_type();
    if (device_type == DeviceType::kDeviceUnknown) 
    {
        LOG(ERROR) << "The device type of the tensor is unknown.";
    }
    else if (device_type == DeviceType::kDeviceCPU) 
    {
        size_t byte_size = this->byte_size();
        auto cu_alloc = CUDADeviceAllocatorFactory::get_instance();
        auto cu_buffer = std::make_shared<Buffer>(byte_size, cu_alloc);
        cu_alloc->memcpy(buffer_->ptr(), cu_buffer->ptr(), byte_size, MemcpyKind::kMemcpyCPU2CUDA, stream);
        this->buffer_ = cu_buffer;
    } 
    else 
    {
        LOG(INFO) << "The device type of the tensor is already cpu.";
    }
}

bool Tensor::is_empty() const
{
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(std::shared_ptr<DeviceAllocator> alloc, DataType data_type, bool need_alloc, void* ptr) 
{
    if (!alloc && !need_alloc) 
    {
        std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(data_type_size(data_type) * size_, nullptr, ptr, true);
        this->buffer_ = buffer;
    } 
    else 
    {
        allocate(alloc, true);
    }
}
}