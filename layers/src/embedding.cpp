#include "layer.h"
#include "embedding.h"
// #include "cpu/emb_kernel.h"
#include "kernels_interface.h"

namespace my_vllm
{

EmbeddingLayer::EmbeddingLayer(DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size)
    : dim_(dim), seq_len_(seq_len), vocab_size_(vocab_size),
    LayerParam(device_type, LayerType::kLayerEmbedding, "Embedding")
{
    reset_weight_size(1);
    reset_input_size(2);
    //input1:input_token input2:token_num
    reset_output_size(1);
}

Status EmbeddingLayer::check() const
{
    const auto& input_tensor = get_input(0);
    const auto& token_size = get_input(1).size();
    if (token_size > static_cast<size_t>(seq_len_) || token_size != input_tensor.size())
    {
        return InvalidArgument("The input token count is invalid for the model context window.");
    }

    const int32_t token_count = static_cast<int32_t>(token_size);
    Status status = check_tensor_with_dim(input_tensor, DeviceType::kDeviceCPU,
                                          DataType::kDataTypeInt32, token_count);
    if (!status)
    {
        LOG(ERROR) << "The input tensor error in the embedding layer.";
        return status;
    }

    for (size_t i = 0; i < token_size; ++i)
    {
        const int32_t token = input_tensor.ptr<int32_t>()[i];
        if (token < 0 || token >= vocab_size_)
        {
            return InvalidArgument("An input token id is outside the embedding vocabulary.");
        }
    }

    status = check_tensor_with_dim(get_weight(0), device_type_, data_type_, vocab_size_, dim_);
    if (!status)
    {
        LOG(ERROR) << "The weight tensor error in the embedding layer.";
        return status;
    }

    status = check_tensor_with_dim(get_output(0), device_type_, data_type_, token_count, dim_);
    if (!status)
    {
        LOG(ERROR) << "The output tensor error in the embedding layer.";
        return status;
    }
    return Success();
}

Status EmbeddingLayer::forward()
{
    Status status = check();
    if (!status)
    {
        return status;
    }
    if (device_type_ == DeviceType::kDeviceCUDA)
    {
        CHECK(cuda_config_ != nullptr);
    }
    get_emb_kernel(device_type_)(get_input(0), get_weight(0), get_output(0), vocab_size_, cuda_config_ ? cuda_config_->stream : nullptr);
    return StatusCode::kSuccess;
}
}
