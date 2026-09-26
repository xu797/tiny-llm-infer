#include "tensor.h"

#include <limits>
#include "cuda_alloc.h"
#include "cpu_alloc.h"

namespace my_vllm
{

template <typename T>
static size_t reduce_dimension(T begin, T end, size_t init)
{
    if (begin >= end)
    {
        return 0;
    }

    size_t size = init;
    for (T dimension = begin; dimension != end; ++dimension)
    {
        if (*dimension <= 0)
        {
            return 0;
        }
        const size_t value = static_cast<size_t>(*dimension);
        if (size > std::numeric_limits<size_t>::max() / value)
        {
            return 0;
        }
        size *= value;
    }
    return size;
}

Tensor::Tensor(DataType data_type, int32_t dim0, bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void *ptr)
        : data_type_(data_type)
{
    dims_.push_back(dim0);
    size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
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

Tensor::Tensor(
    DataType data_type, int32_t dim0, int32_t dim1, 
    bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void *ptr)
    : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
    if (need_alloc && alloc)
    {
        allocate(alloc);
    }
    else 
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(
    DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, 
    bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void *ptr) 
    : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
    if (need_alloc && alloc) 
    {
        allocate(alloc);
    } 
    else 
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(
    DataType data_type, int32_t dim0, int32_t dim1, int32_t dim2, int32_t dim3, 
    bool need_alloc, std::shared_ptr<DeviceAllocator> alloc, void *ptr) 
    : data_type_(data_type) 
{
    dims_.push_back(dim0);
    dims_.push_back(dim1);
    dims_.push_back(dim2);
    dims_.push_back(dim3);
    size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
    if (need_alloc && alloc)
    {
        allocate(alloc);
    } 
    else
    {
        init_buffer(alloc, data_type_, need_alloc, ptr);
    }
}

Tensor::Tensor(
    DataType data_type, std::vector<int32_t> dims, bool need_alloc,
    std::shared_ptr<DeviceAllocator> alloc, void *ptr)
    : dims_(std::move(dims)), data_type_(data_type)
{
    size_ = reduce_dimension(dims_.begin(), dims_.end(), size_t{1});
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

static size_t data_type_size(DataType data_type)
{
    switch (data_type) 
    {
        case DataType::kDataTypeFp32: 
        {
            return 4;
        }
        case DataType::kDataTypeInt32: 
        {
            return 4;
        }
        default: 
        {
            LOG(FATAL) << "Unknown data type size for " << int(data_type);
            return 0;
        }
    }
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

void Tensor::reshape(const std::vector<int32_t>& dims) 
{
    size_t size = reduce_dimension(dims.begin(), dims.end(), size_t{1});
    if (!buffer_)
    {
        this->dims_ = dims;
        this->size_ = size;
        return;
    }

    if (size > size_) 
    {
        auto new_buffer = std::make_shared<Buffer>(size * DataTypeSize(this->data_type_), buffer_->allocator());
        CHECK(new_buffer->allocate());
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }
    this->dims_ = dims;
    this->size_ = size;
}

std::shared_ptr<Buffer> Tensor::get_buffer() const 
{
    return buffer_; 
}

Tensor Tensor::clone() const 
{
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();

    auto allocator = buffer_->allocator();
    new_tensor.buffer_ = std::make_shared<Buffer>(byte_size, allocator);
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
}

size_t Tensor::byte_size() const 
{ 
    return this->size() * DataTypeSize(data_type_); 
}

std::vector<size_t> Tensor::strides() const 
{
    std::vector<size_t> strides;
    if (!dims_.empty())
    {
        for (int32_t i = 0; i < dims_.size() - 1; ++i)
        {
            size_t stride = reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
            strides.push_back(stride);
        }
        strides.push_back(1);
    }
    return strides;
}

const std::vector<int32_t>& Tensor::dims() const 
{ 
    return this->dims_;
}

void Tensor::set_device_type(DeviceType device_type) const 
{
    if (buffer_) 
    {
        buffer_->set_device_type(device_type);
    }
}

void Tensor::reset(DataType data_type, const std::vector<int32_t>& dims) 
{
    this->data_type_ = data_type;
    this->dims_ = dims;
    this->size_ = reduce_dimension(dims.begin(), dims.end(), 1);
    this->buffer_ = nullptr;
}

DeviceType Tensor::device_type() const 
{
    if (!buffer_) 
    {
    return DeviceType::kDeviceUnknown;
    }
    return buffer_->device_type();
}

bool Tensor::assign(std::shared_ptr<Buffer> buffer) 
{
    if (!buffer) 
    {
        LOG(ERROR) << "The buffer parameter in the assign function is null pointer!";
        return false;
    }
    if (buffer_) 
    {
        if (buffer_->device_type() != buffer->device_type()) {
            LOG(ERROR) << "The device type of the new buffer is different from the original one.";
        }
    }

    size_t byte_size = this->byte_size();
    if (byte_size > buffer->byte_size())
    {
        LOG(ERROR) << "The size of buffer is too small for the tensor!";
        return false;
    }
    buffer_ = buffer;
    return true;
}

int32_t Tensor::dims_size() const 
{ 
    return static_cast<int32_t>(dims_.size()); 
}

DataType Tensor::data_type() const 
{ 
    return data_type_; 
}

size_t Tensor::size() const 
{ 
    return this->size_; 
}

int32_t Tensor::get_dim(int32_t idx) const 
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, this->dims_.size());
    return this->dims_.at(idx);
}

}