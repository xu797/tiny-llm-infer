#include <cstdarg>
#include <numeric>
#include <utility>
#include <glog/logging.h>

#include "layer.h"
// #include "cuda_config.h"

namespace my_vllm
{

BaseLayer::BaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type, std::string layer_name)
    : device_type_(device_type), layer_type_(layer_type), data_type_(data_type), layer_name_(std::move(layer_name))
{

}

DataType BaseLayer::data_type() const
{
    return data_type_;
}

LayerType BaseLayer::layer_type() const
{
    return layer_type_;
}

Status BaseLayer::set_weight(int32_t idx, const Tensor& weight)
{
    return FunctionNotImplement();
}

Status BaseLayer::set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                   const void* weight_ptr, DeviceType device_type)
{
    return FunctionNotImplement();
}

const std::string& BaseLayer::get_layer_name() const
{
    return layer_name_;
}

void BaseLayer::set_layer_name(const std::string& layer_name)
{
    layer_name_ = layer_name;
}

DeviceType BaseLayer::device_type() const
{
    return device_type_;
}

void BaseLayer::set_device_type(DeviceType device_type)
{
    device_type_ = device_type;
}

Layer::Layer(DeviceType device_type, LayerType layer_type, std::string layer_name)
    : BaseLayer(device_type, layer_type, DataType::kDataTypeFp32, std::move(layer_name))
{

}

Status Layer::init()
{
    return Success();
}

Status Layer::forward()
{
    return FunctionNotImplement("");
}

Status Layer::check_tensor(const Tensor& tensor, DeviceType device_type,
                                 DataType data_type) const
{
    if (tensor.is_empty())
    {
        return InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type)
    {
        return InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type)
    {
        return InvalidArgument("The tensor has a wrong data type.");
    }
    return Success();
}

Status Layer::check_tensor_with_dim(const Tensor& tensor, DeviceType device_type, DataType data_type, ...) const
{
    std::va_list args;
    if (tensor.is_empty())
    {
        return InvalidArgument("The tensor parameter is empty.");
    }
    if (tensor.device_type() != device_type)
    {
        return InvalidArgument("The tensor has a wrong device type.");
    }
    if (tensor.data_type() != data_type)
    {
        return InvalidArgument("The tensor has a wrong data type.");
    }

    va_start(args, data_type);
    int32_t dims = tensor.dims_size();
    for (int32_t i = 0; i < dims; ++i)
    {
        int32_t dim = va_arg(args, int32_t);
        if (dim != tensor.get_dim(i))
        {
            return InvalidArgument("The tensor has a wrong dim in dim" + std::to_string(i));
        }
    }
    va_end(args);
    return Success();
}

void Layer::set_input(int32_t idx, const Tensor& input)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    this->inputs_.at(idx) = input;
}

void Layer::set_output(int32_t idx, const Tensor& output)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    this->outputs_.at(idx) = output;
}

const Tensor& Layer::get_input(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

Tensor& Layer::get_input(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, inputs_.size());
    return inputs_.at(idx);
}

Tensor& Layer::get_output(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

Status Layer::check() const
{
    return FunctionNotImplement("The check function is not implement yet");
}

const Tensor& Layer::get_output(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, outputs_.size());
    return outputs_.at(idx);
}

void Layer::reset_input_size(size_t size)
{
    inputs_.resize(size);
}

void Layer::reset_output_size(size_t size)
{
    outputs_.resize(size);
}

void Layer::to_cuda()
{
    for (auto& input : inputs_)
    {
        if (!input.is_empty())
        {
            input.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
    for (auto& output : outputs_)
    {
        if (!output.is_empty())
        {
            output.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
        }
    }
}

void Layer::set_cuda_config(std::shared_ptr<CudaConfig> config)
{
    if (!config)
    {
        return;
    }
    this->cuda_config_ = config;
}

std::shared_ptr<CudaConfig> Layer::cuda_config() const
{
    return cuda_config_;
}

size_t Layer::input_size() const
{
    return inputs_.size();
}

size_t Layer::output_size() const
{
    return outputs_.size();
}

LayerParam::LayerParam(DeviceType device_type, LayerType layer_type, bool is_quant_layer, std::string layer_name)
    : Layer(device_type, layer_type, std::move(layer_name)), is_quant_layer_(is_quant_layer)
{

}

Status LayerParam::set_weight(int32_t idx, const Tensor& weight)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK(weight.data_type() == DataType::kDataTypeFp32);
    if (!weight.is_empty())
    {
        CHECK(weight.device_type() == device_type_);
    }
    weights_.at(idx) = weight;
    return Success();
}

const Tensor& LayerParam::get_weight(int32_t idx) const
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

void LayerParam::to_cuda()
{
    Layer::to_cuda();
    for (auto& weight : weights_)
    {
        weight.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
    if (!scales_.is_empty())
    {
        scales_.to_cuda(cuda_config_ ? cuda_config_->stream : nullptr);
    }
}

Status LayerParam::set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr, DeviceType device_type)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    CHECK_NE(weight_ptr, nullptr);

    size_t size = std::accumulate(dims.begin(), dims.end(), sizeof(float), std::multiplies<>());
    std::shared_ptr<Buffer> buffer = std::make_shared<Buffer>(size, nullptr, const_cast<void*>(weight_ptr), true);
    if (device_type != DeviceType::kDeviceUnknown)
    {
        buffer->set_device_type(device_type);
    }

    if (!is_quant_layer_)
    {
        Tensor weight(DataType::kDataTypeFp32, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        weights_.at(idx) = weight;
    }
    else
    {
        // is quant layer
        Tensor weight(DataType::kDataTypeInt8, dims);
        weight.set_device_type(device_type);
        CHECK(weight.assign(buffer));
        weights_.at(idx) = weight;

        const int32_t weight_size = static_cast<int32_t>(weight.size());
        CHECK(weight_size % group_size_ == 0);

        int32_t scale_nums = weight_size / group_size_;
        scales_ = Tensor{DataType::kDataTypeFp32, scale_nums, false, nullptr,
                                    reinterpret_cast<float*>((int8_t*)weight_ptr + weight_size)};
        scales_.set_device_type(device_type);
    }

    return Success();
}

void LayerParam::set_scales(const Tensor& scales)
{
    CHECK(!scales.is_empty());
    this->scales_ = scales;
}

void LayerParam::set_group_size(int32_t group_size)
{
    this->group_size_ = group_size;
}


int32_t LayerParam::get_scale_num() const
{
    CHECK(!scales_.is_empty());
    return static_cast<int32_t>(scales_.size());
}

void LayerParam::reset_weight_size(size_t size)
{
    weights_.resize(size);
}

size_t LayerParam::weight_size() const
{
    return weights_.size();
}

Status Layer::forward(const Tensor& input1, const Tensor& output1)
{
    this->set_input(0, input1);
    this->set_output(0, output1);
    return this->forward();
}

Status Layer::forward(const Tensor& input1, const Tensor& input2, const Tensor& output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);

    this->set_output(0, output1);
    return this->forward();
}

Status Layer::forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);

    this->set_output(0, output1);
    return this->forward();
}

Status Layer::forward(const Tensor& input1, const Tensor& input2, const Tensor& input3, const Tensor& input4, const Tensor& output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);

    this->set_output(0, output1);
    return this->forward();
}

Status Layer::forward(const Tensor& input1, const Tensor& input2,
                            const Tensor& input3, const Tensor& input4,
                            const Tensor& input5, const Tensor& output1)
{
    this->set_input(0, input1);
    this->set_input(1, input2);
    this->set_input(2, input3);
    this->set_input(3, input4);
    this->set_input(4, input5);

    this->set_output(0, output1);
    return this->forward();
}

Tensor& LayerParam::get_weight(int32_t idx)
{
    CHECK_GE(idx, 0);
    CHECK_LT(idx, weights_.size());
    return weights_.at(idx);
}

}  
