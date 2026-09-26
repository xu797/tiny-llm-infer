#include <glog/logging.h>
#include <fstream>
#include <limits>
#include <stdexcept>

#include "encode.h"
#include "unicode.h"
#include "nlohmann/json.hpp"

namespace my_vllm
{

#if defined (LLAMA3_SUPPORT) || defined (QWEN2_SUPPORT)
static const std::string PAT_STR =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?:$|[^\S])|\s+)";

BpeEncodeLayer::BpeEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);

  json data = json::parse(f);
  const auto& datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }
  bos_id_ = special_tokens["<|begin_of_text|>"];
  eos_id_ = special_tokens["<|end_of_text|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|eot_id|>"];

  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}

std::vector<int32_t> BpeEncodeLayer::encode(const std::string& sentence) const {
  CHECK(this->tiktoken_ != nullptr);
  std::map<std::string, std::string> replacements;
  replacements[" "] = "Ġ";
  std::string s = absl::StrReplaceAll(sentence, replacements);
  auto input_ids = this->tiktoken_->encode(s);

  if (has_bos_) {
    input_ids.insert(input_ids.begin(), bos_id_);
  }
  if (has_eos_) {
    input_ids.push_back(eos_id_);
  }
  return input_ids;
}

std::string BpeEncodeLayer::decode(int32_t token_id) const { return ""; }

std::string BpeEncodeLayer::decode(const std::vector<int32_t>& token_ids) const {
  CHECK(this->tiktoken_ != nullptr);
  auto s = tiktoken_->decode(token_ids);
  std::map<std::string, std::string> reverse_replacements;
  reverse_replacements["Ġ"] = " ";
  const std::string& sentence = absl::StrReplaceAll(s, reverse_replacements);
  return sentence;
}

bool BpeEncodeLayer::is_sentence_ending(int32_t token_id) const {
  if (token_id == stop_token1_ || token_id == stop_token2_) {
    return true;
  } else {
    return false;
  }
}

int32_t BpeEncodeLayer::vocab_size() const {
  CHECK(this->tiktoken_ != nullptr);
  return num_token_;
}

QwenEncodeLayer::QwenEncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : BpeEncodeLayer(std::move(token_model_path), has_bos, has_eos) {
  using json = nlohmann::json;
  std::ifstream f(token_model_path_);

  json data = json::parse(f);
  const auto& datas = data["added_tokens"];
  ankerl::unordered_dense::map<std::string, int> special_tokens;
  for (const auto& data1 : datas) {
    int id = data1["id"];
    std::string content = data1["content"];
    special_tokens.insert({content, id});
  }

  ankerl::unordered_dense::map<std::string, int> encoder;
  const auto& vocabs = data["model"]["vocab"];
  const auto& vocab_items = vocabs.items();
  for (const auto& v : vocab_items) {
    const auto cpts = unicode_cpts_from_utf8(v.key());
    std::string key;
    for (const auto cpt : cpts) {
        const auto utf8 = unicode_cpt_to_utf8(cpt);
        key += unicode_utf8_to_byte(utf8);
    }
    const int32_t id = v.value();
    encoder[key] = id;
  }
  bos_id_ = special_tokens["<|im_start|>"];
  eos_id_ = special_tokens["<|im_end|>"];
  stop_token1_ = eos_id_;
  stop_token2_ = special_tokens["<|endoftext|>"];

  num_token_ = encoder.size() + special_tokens.size();
  tiktoken_ = std::make_unique<tiktoken::tiktoken>(encoder, special_tokens, PAT_STR);
}


#endif

Qwen3EncodeLayer::Qwen3EncodeLayer(std::string token_model_path, bool has_bos, bool has_eos)
    : EncodeLayerBase(std::move(token_model_path), has_bos, has_eos)
{
    std::ifstream input(token_model_path_);
    if (!input)
    {
        throw std::runtime_error("Cannot open Qwen3 tokenizer: " + token_model_path_);
    }
    const auto json = nlohmann::json::parse(input);
    if (!json.contains("model") || !json["model"].contains("vocab") ||
        !json["model"].contains("merges"))
    {
        throw std::runtime_error("Qwen3 tokenizer.json is missing its BPE vocabulary or merges.");
    }

    for (auto item = json["model"]["vocab"].begin(); item != json["model"]["vocab"].end(); ++item)
    {
        const int32_t id = item.value().get<int32_t>();
        encoder_.emplace(item.key(), id);
        decoder_.emplace(id, item.key());
        num_token_ = std::max(num_token_, id + 1);
    }
    int32_t rank = 0;
    for (const auto& merge : json["model"]["merges"])
    {
        if (!merge.is_array() || merge.size() != 2)
        {
            throw std::runtime_error("Unexpected Qwen3 BPE merge format in tokenizer.json.");
        }
        merge_ranks_.emplace(merge[0].get<std::string>() + " " +
                                 merge[1].get<std::string>(),
                             rank++);
    }

    if (json.contains("added_tokens"))
    {
        for (const auto& token : json["added_tokens"])
        {
            const int32_t id = token.at("id").get<int32_t>();
            const std::string content = token.at("content").get<std::string>();
            special_encoder_[content] = id;
            special_decoder_[id] = content;
            special_tokens_.push_back(content);
            num_token_ = std::max(num_token_, id + 1);
        }
    }
    std::sort(special_tokens_.begin(), special_tokens_.end(),
              [](const std::string& lhs, const std::string& rhs) {
                  return lhs.size() > rhs.size();
              });

    const auto endoftext = special_encoder_.find("<|endoftext|>");
    const auto eos = special_encoder_.find("<|im_end|>");
    endoftext_id_ = endoftext == special_encoder_.end() ? -1 : endoftext->second;
    eos_id_ = eos == special_encoder_.end() ? endoftext_id_ : eos->second;
    if (has_bos_ && endoftext_id_ < 0)
    {
        throw std::runtime_error("Qwen3 tokenizer does not define <|endoftext|> as its BOS token.");
    }
}

std::vector<int32_t> Qwen3EncodeLayer::encode_piece(const std::string& piece) const
{
    if (piece.empty()) return {};

    std::vector<std::string> symbols;
    for (const uint32_t codepoint : unicode_cpts_from_utf8(piece))
    {
        symbols.push_back(unicode_cpt_to_utf8(codepoint));
    }
    while (symbols.size() > 1)
    {
        int32_t best_rank = std::numeric_limits<int32_t>::max();
        size_t best_index = symbols.size();
        for (size_t i = 0; i + 1 < symbols.size(); ++i)
        {
            const auto found = merge_ranks_.find(symbols[i] + " " + symbols[i + 1]);
            if (found != merge_ranks_.end() && found->second < best_rank)
            {
                best_rank = found->second;
                best_index = i;
            }
        }
        if (best_index == symbols.size()) break;
        symbols[best_index] += symbols[best_index + 1];
        symbols.erase(symbols.begin() + static_cast<std::ptrdiff_t>(best_index + 1));
    }

    std::vector<int32_t> ids;
    ids.reserve(symbols.size());
    for (const std::string& symbol : symbols)
    {
        const auto found = encoder_.find(symbol);
        if (found == encoder_.end())
        {
            throw std::runtime_error("Qwen3 BPE produced a token missing from tokenizer vocabulary.");
        }
        ids.push_back(found->second);
    }
    return ids;
}

void Qwen3EncodeLayer::encode_ordinary(const std::string& text,
                                       std::vector<int32_t>& output) const
{
    static const std::vector<std::string> pattern = {
        "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"};
    for (const std::string& piece : unicode_regex_split(text, pattern))
    {
        auto ids = encode_piece(piece);
        output.insert(output.end(), ids.begin(), ids.end());
    }
}

std::vector<int32_t> Qwen3EncodeLayer::encode(const std::string& sentence) const
{
    std::vector<int32_t> ids;
    if (has_bos_) ids.push_back(endoftext_id_);

    size_t cursor = 0;
    while (cursor < sentence.size())
    {
        size_t next_special = std::string::npos;
        const std::string* special_text = nullptr;
        for (const auto& token : special_tokens_)
        {
            const size_t found = sentence.find(token, cursor);
            if (found < next_special)
            {
                next_special = found;
                special_text = &token;
            }
        }
        if (!special_text)
        {
            encode_ordinary(sentence.substr(cursor), ids);
            break;
        }
        if (next_special > cursor)
        {
            encode_ordinary(sentence.substr(cursor, next_special - cursor), ids);
        }
        ids.push_back(special_encoder_.at(*special_text));
        cursor = next_special + special_text->size();
    }
    if (has_eos_) ids.push_back(eos_id_);
    return ids;
}

std::string Qwen3EncodeLayer::decode(int32_t token_id) const
{
    return decode(std::vector<int32_t>{token_id});
}

std::string Qwen3EncodeLayer::decode(const std::vector<int32_t>& token_ids) const
{
    std::string result;
    for (const int32_t id : token_ids)
    {
        const auto special = special_decoder_.find(id);
        if (special != special_decoder_.end())
        {
            result += special->second;
            continue;
        }
        const auto token = decoder_.find(id);
        if (token == decoder_.end())
        {
            throw std::runtime_error("Unknown Qwen3 token id: " + std::to_string(id));
        }
        for (const uint32_t codepoint : unicode_cpts_from_utf8(token->second))
        {
            result.push_back(static_cast<char>(unicode_utf8_to_byte(unicode_cpt_to_utf8(codepoint))));
        }
    }
    return result;
}

bool Qwen3EncodeLayer::is_sentence_ending(int32_t token_id) const
{
    return token_id == eos_id_ || token_id == endoftext_id_;
}

int32_t Qwen3EncodeLayer::vocab_size() const
{
    return num_token_;
}

}  
