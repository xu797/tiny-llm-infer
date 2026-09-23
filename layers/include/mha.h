#ifndef MYVLLM_LAYERS_MHA_H_
#define MYVLLM_LAYERS_MHA_H_

#include "cuda_config.h"
#include "layer.h"

namespace my_vllm
{
class MultiHeadAttention : public Layer 
{
public:
    explicit MultiHeadAttention(DeviceType device_type, int32_t layer_index,
                                int32_t kv_mul, int32_t kv_dim, int32_t seq_len,
                                int32_t head_num, int32_t head_size);

    Status check() const override;

    void set_pos(int32_t pos);
    void set_layer_idx(int32_t layer_idx);

    Status forward() override;

private:
    int32_t layer_index_ = 0;
    int32_t pos_ = 0;
    int32_t kv_mul_ = 0;
    int32_t kv_dim_ = 0;
    int32_t seq_len_ = 0;
    int32_t head_num_ = 0;
    int32_t head_size_ = 0;
};
}

#endif