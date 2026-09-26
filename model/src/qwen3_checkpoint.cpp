#include "qwen3.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <nlohmann/json.hpp>

namespace my_vllm
{
Status Qwen3Model::read_qwen3_config(ModelConfig& config)
{
    using json = nlohmann::json;
    const std::filesystem::path config_path =
        std::filesystem::path(model_path_).parent_path() / "config.json";
    std::ifstream input(config_path);
    if (!input)
        return PathNotValid("Cannot open the Qwen3 config file: " + config_path.string());

    try
    {
        const json data = json::parse(input);
        if (data.value("model_type", std::string()) != "qwen3")
            return ModelParseError("The config.json next to the weights is not a Qwen3 model.");

        qwen_head_dim_ = data.value("head_dim", 0);
        rms_norm_eps_ = data.value("rms_norm_eps", 1e-6f);
        rope_theta_ = data.value("rope_theta", 1000000.0f);
        config.dim = data.at("hidden_size").get<int32_t>();
        config.hidden_dim = data.at("intermediate_size").get<int32_t>();
        config.layer_num = data.at("num_hidden_layers").get<int32_t>();
        config.head_num = data.at("num_attention_heads").get<int32_t>();
        config.kv_head_num = data.at("num_key_value_heads").get<int32_t>();
        config.vocab_size = data.at("vocab_size").get<int32_t>();
        const int32_t model_context = data.at("max_position_embeddings").get<int32_t>();
        config.seq_len = std::min(requested_seq_len_, model_context);
    }
    catch (const std::exception& error)
    {
        return ModelParseError(std::string("Failed to parse Qwen3 config.json: ") +
                               error.what());
    }
    return Success();
}

Status Qwen3Model::map_safetensors_file()
{
    safetensors_fd_ = open(model_path_.c_str(), O_RDONLY);
    if (safetensors_fd_ < 0)
        return PathNotValid("Cannot open the Qwen3 safetensors file: " + model_path_);

    struct stat file_stat {};
    if (fstat(safetensors_fd_, &file_stat) != 0 || file_stat.st_size < 8)
    {
        release_safetensors_map();
        return ModelParseError("The Qwen3 safetensors file is too small or cannot be read.");
    }

    safetensors_file_size_ = static_cast<size_t>(file_stat.st_size);
    safetensors_mapping_ = mmap(nullptr, safetensors_file_size_, PROT_READ, MAP_PRIVATE,
                                safetensors_fd_, 0);
    if (safetensors_mapping_ == MAP_FAILED || safetensors_mapping_ == nullptr)
    {
        safetensors_mapping_ = nullptr;
        release_safetensors_map();
        return ModelParseError("Failed to map the Qwen3 safetensors file into memory.");
    }

    uint64_t header_size = 0;
    std::memcpy(&header_size, safetensors_mapping_, sizeof(header_size));
    if (header_size > safetensors_file_size_ - sizeof(header_size))
    {
        release_safetensors_map();
        return ModelParseError("The Qwen3 safetensors header size is invalid.");
    }

    safetensors_data_start_ = sizeof(header_size) + static_cast<size_t>(header_size);
    try
    {
        using json = nlohmann::json;
        const char* header_data =
            static_cast<const char*>(safetensors_mapping_) + sizeof(header_size);
        const json header = json::parse(
            std::string(header_data, static_cast<size_t>(header_size)));
        const size_t data_size = safetensors_file_size_ - safetensors_data_start_;
        for (auto item = header.begin(); item != header.end(); ++item)
        {
            if (item.key() == "__metadata__") continue;

            SafeTensorInfo info;
            info.dtype = item.value().at("dtype").get<std::string>();
            for (const auto& dimension : item.value().at("shape"))
                info.shape.push_back(dimension.get<int32_t>());
            const auto offsets = item.value().at("data_offsets");
            info.begin = offsets.at(0).get<size_t>();
            info.end = offsets.at(1).get<size_t>();
            if (info.begin > info.end || info.end > data_size)
            {
                release_safetensors_map();
                return ModelParseError("A Qwen3 tensor has an invalid safetensors data offset.");
            }
            safetensors_.emplace(item.key(), std::move(info));
        }
    }
    catch (const std::exception& error)
    {
        release_safetensors_map();
        return ModelParseError(std::string("Failed to parse Qwen3 safetensors header: ") +
                               error.what());
    }
    return Success();
}

Status Qwen3Model::read_model_file()
{
    ModelConfig config;
    Status status = read_qwen3_config(config);
    if (!status) return status;

    status = map_safetensors_file();
    if (!status) return status;

    status = generate_model_infos(config);
    if (!status) release_safetensors_map();
    return status;
}

Status Qwen3Model::generate_model_infos(const ModelConfig& config) const
{
    if (qwen_head_dim_ <= 0 || config.head_num <= 0 || config.kv_head_num <= 0 ||
        config.vocab_size <= 0 || config.seq_len <= 0 ||
        config.kv_head_num > config.head_num ||
        config.head_num % config.kv_head_num != 0 || qwen_head_dim_ % 2 != 0)
        return ModelParseError("The Qwen3 config.json contains invalid attention dimensions.");

    Status status = Model::generate_model_infos(config);
    if (!status) return status;
    config_->head_size_ = qwen_head_dim_;
    config_->kv_dim_ = config.kv_head_num * qwen_head_dim_;
    config_->kv_mul_ = config.head_num / config.kv_head_num;
    config_->query_dim_ = config.head_num * qwen_head_dim_;
    return Success();
}

Status Qwen3Model::create_encode_layer()
{
    try
    {
        encode_layer_ = std::make_unique<Qwen3EncodeLayer>(token_path_, false, false);
    }
    catch (const std::exception& error)
    {
        return ModelParseError(std::string("Failed to create Qwen3 tokenizer: ") + error.what());
    }
    if (!encode_layer_ || encode_layer_->vocab_size() <= 0)
        return InternalError("The Qwen3 tokenizer vocabulary is empty.");
    return Success();
}

void Qwen3Model::release_safetensors_map()
{
    if (safetensors_mapping_ != nullptr && safetensors_mapping_ != MAP_FAILED)
    {
        munmap(safetensors_mapping_, safetensors_file_size_);
        safetensors_mapping_ = nullptr;
    }
    if (safetensors_fd_ >= 0)
    {
        close(safetensors_fd_);
        safetensors_fd_ = -1;
    }
    safetensors_file_size_ = 0;
    safetensors_data_start_ = 0;
    safetensors_.clear();
}
}  // namespace my_vllm
