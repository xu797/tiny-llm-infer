#ifndef MYVLLM_LAYERS_EMBEDDING_H_
#define MYVLLM_LAYERS_EMBEDDING_H_

#include <utility>

#include "layer.h"

namespace my_vllm
{
struct EmbeddingOutput 
{
    Tensor input_tokens;
    Tensor input_embeddings;
    Tensor input_token_num;
    explicit EmbeddingOutput(Tensor input_tokens, Tensor input_embeddings, Tensor input_token_num)
        : input_tokens(std::move(input_tokens)),
        input_embeddings(std::move(input_embeddings)),
        input_token_num(std::move(input_token_num)) {}
};

class EmbeddingLayer : public LayerParam 
{
public:
    explicit EmbeddingLayer(DeviceType device_type, int32_t dim, int32_t seq_len, int32_t vocab_size);

    Status check() const override;

    Status forward() override;

private:
    int32_t dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t vocab_size_ = 0;
};
}

#endif