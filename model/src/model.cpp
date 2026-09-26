#include "model.h"

namespace my_vllm
{
    Model::Model(ModelType model_type, std::string token_path, std::string model_path)
        : model_type_(model_type),
          token_path_(std::move(token_path)),
          model_path_(std::move(model_path))
    {
    }

    ModelType Model::model_type() const
    {
        return model_type_;
    }

    const std::string &Model::token_path() const
    {
        return token_path_;
    }

    const std::string &Model::model_path() const
    {
        return model_path_;
    }

    Status Model::insert_buffer(ModelBufferType buffer_idx, const Tensor &tensor)
    {
        if (buffers_.count(buffer_idx) > 0)
        {
            return KeyHasExits(std::to_string(int(buffer_idx)) + " has exits in the buffers");
        }
        if (tensor.is_empty())
        {
            return InvalidArgument("The tensor is empty for inserting buffer.");
        }
        buffers_.insert({buffer_idx, tensor});
        return Success();
    }

    Tensor &Model::get_buffer(ModelBufferType buffer_idx)
    {
        CHECK_GT(buffers_.count(buffer_idx), 0) << int(buffer_idx);
        return buffers_.at(buffer_idx);
    }

    const Tensor &Model::get_buffer(ModelBufferType buffer_idx) const
    {
        CHECK_GT(buffers_.count(buffer_idx), 0);
        return buffers_.at(buffer_idx);
    }

    Status Model::generate_model_infos(const ModelConfig& config) const
    {
        if (config.dim <= 0 || config.hidden_dim <= 0 || config.layer_num <= 0 ||
            config.head_num <= 0 || config.kv_head_num <= 0 || config.seq_len <= 0 ||
            config.vocab_size <= 0 || config.dim % config.head_num != 0 ||
            config.head_num % config.kv_head_num != 0 ||
            (config.dim / config.head_num) % 2 != 0)
        {
            return ModelParseError("The Qwen3 config contains invalid transformer dimensions.");
        }

        config_->dim_ = config.dim;
        config_->hidden_dim_ = config.hidden_dim;
        config_->layer_num_ = config.layer_num;
        config_->head_num_ = config.head_num;
        config_->kv_head_num_ = config.kv_head_num;
        config_->seq_len_ = config.seq_len;
        config_->kv_dim_ = (config.dim * config.kv_head_num) / config.head_num;
        config_->kv_mul_ = config.head_num / config.kv_head_num;
        config_->head_size_ = config.dim / config.head_num;
        config_->query_dim_ = config.dim;
        config_->vocab_size_ = config.vocab_size;
        return Success();
    }

    Status Model::gen_model_from_file()
    {
        //
        config_ = std::make_unique<TransformerConfig>();

        // init sentence piece processor
        // google sentence piece
        auto create_encode_status = create_encode_layer();
        if (!create_encode_status)
        {
            LOG(ERROR) << "Create the encode layer failed!";
            return create_encode_status;
        }
        // mmap
        auto mmap_status = read_model_file();
        if (!mmap_status)
        {
            LOG(ERROR) << "Handle model file " << model_path_ << " failed!";
            return mmap_status;
        }
        auto layer_create_status = create_layers();
        if (!layer_create_status)
        {
            LOG(ERROR) << "Create layers for the model file " << model_path_ << " failed!";
            return layer_create_status;
        }

        return Success();
    }

}
