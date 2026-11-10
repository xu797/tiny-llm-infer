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
    const std::filesystem::path supplied_model_path(model_path_);
    const std::filesystem::path config_dir = std::filesystem::is_directory(supplied_model_path)
        ? supplied_model_path : supplied_model_path.parent_path();
    const std::filesystem::path config_path = config_dir / "config.json";
    std::ifstream input(config_path);
    if (!input)
        return PathNotValid("Cannot open the Qwen3 config file: " + config_path.string());

    try
    {
        const json data = json::parse(input);
        const std::string model_type = data.value("model_type", std::string());
        if (model_type != "qwen3" && model_type != "qwen2")
            return ModelParseError("Only Qwen2 and Qwen3 model configs are supported.");
        is_qwen3_ = model_type == "qwen3";

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
        if (qwen_head_dim_ == 0) qwen_head_dim_ = config.dim / config.head_num;
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
    namespace fs = std::filesystem;
    const fs::path supplied(model_path_);
    const fs::path model_dir = fs::is_directory(supplied) ? supplied : supplied.parent_path();
    fs::path index_path = model_dir / "model.safetensors.index.json";
    std::unordered_map<std::string, std::string> tensor_files;
    if (fs::exists(index_path))
    {
        try
        {
            std::ifstream index_input(index_path);
            const auto index = nlohmann::json::parse(index_input);
            for (auto item = index.at("weight_map").begin(); item != index.at("weight_map").end(); ++item)
                tensor_files.emplace(item.key(), item.value().get<std::string>());
        }
        catch (const std::exception& error)
        {
            return ModelParseError(std::string("Failed to parse safetensors shard index: ") + error.what());
        }
    }
    std::vector<std::string> filenames;
    if (!tensor_files.empty())
    {
        for (const auto& entry : tensor_files)
            if (std::find(filenames.begin(), filenames.end(), entry.second) == filenames.end())
                filenames.push_back(entry.second);
    }
    else if (fs::is_directory(supplied))
    {
        return PathNotValid("The model directory does not contain model.safetensors.index.json.");
    }
    else
    {
        filenames.push_back(supplied.filename().string());
    }

    try
    {
        using json = nlohmann::json;
        for (const std::string& filename : filenames)
        {
            const fs::path shard_path = model_dir / filename;
            const int fd = open(shard_path.c_str(), O_RDONLY);
            if (fd < 0) throw std::runtime_error("Cannot open shard " + shard_path.string());
            struct stat file_stat {};
            if (fstat(fd, &file_stat) != 0 || file_stat.st_size < 8)
            {
                close(fd);
                throw std::runtime_error("Invalid safetensors shard " + shard_path.string());
            }
            const size_t file_size = static_cast<size_t>(file_stat.st_size);
            void* mapping = mmap(nullptr, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
            if (mapping == MAP_FAILED || !mapping)
            {
                close(fd);
                throw std::runtime_error("Cannot map shard " + shard_path.string());
            }
            uint64_t header_size = 0;
            std::memcpy(&header_size, mapping, sizeof(header_size));
            if (header_size > file_size - sizeof(header_size))
            {
                munmap(mapping, file_size);
                close(fd);
                throw std::runtime_error("Invalid header in shard " + shard_path.string());
            }
            const size_t data_start = sizeof(header_size) + static_cast<size_t>(header_size);
            safetensors_maps_[filename] = {fd, mapping};
            if (safetensors_mapping_ == nullptr)
            {
                safetensors_fd_ = fd;
                safetensors_mapping_ = mapping;
                safetensors_file_size_ = file_size;
                safetensors_data_start_ = data_start;
            }
            const char* header_data = static_cast<const char*>(mapping) + sizeof(header_size);
            const json header = json::parse(std::string(header_data, static_cast<size_t>(header_size)));
            const size_t data_size = file_size - data_start;
            for (auto item = header.begin(); item != header.end(); ++item)
            {
                if (item.key() == "__metadata__") continue;
                SafeTensorInfo info;
                info.filename = filename;
                info.data_start = data_start;
                info.dtype = item.value().at("dtype").get<std::string>();
                for (const auto& dimension : item.value().at("shape"))
                    info.shape.push_back(dimension.get<int32_t>());
                const auto offsets = item.value().at("data_offsets");
                info.begin = offsets.at(0).get<size_t>();
                info.end = offsets.at(1).get<size_t>();
                if (info.begin > info.end || info.end > data_size)
                    throw std::runtime_error("Invalid tensor offsets in shard " + filename);
                safetensors_.emplace(item.key(), std::move(info));
            }
        }
    }
    catch (const std::exception& error)
    {
        release_safetensors_map();
        return ModelParseError(std::string("Failed to map safetensors weights: ") +
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
        const std::filesystem::path supplied_model_path(model_path_);
        const std::filesystem::path config_dir = std::filesystem::is_directory(supplied_model_path)
            ? supplied_model_path : supplied_model_path.parent_path();
        const std::filesystem::path config_path = config_dir / "config.json";
        std::ifstream config_input(config_path);
        const auto config_json = nlohmann::json::parse(config_input);
        encode_layer_ = std::make_unique<Qwen3EncodeLayer>(
            token_path_, false, false, config_json.value("model_type", std::string()) == "qwen3");
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
    for (const auto& entry : safetensors_maps_)
    {
        struct stat file_stat {};
        if (fstat(entry.second.first, &file_stat) == 0 && file_stat.st_size > 0)
            munmap(entry.second.second, static_cast<size_t>(file_stat.st_size));
        close(entry.second.first);
    }
    safetensors_maps_.clear();
    safetensors_mapping_ = nullptr;
    safetensors_fd_ = -1;
    safetensors_file_size_ = 0;
    safetensors_data_start_ = 0;
    safetensors_.clear();
}
}  // namespace my_vllm
