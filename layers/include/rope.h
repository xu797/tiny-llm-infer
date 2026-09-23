#ifndef MYVLLM_LAYERS_ROPE_H_
#define MYVLLM_LAYERS_ROPE_H_

#include "layer.h"

namespace my_vllm 
{
class RoPELayer : public Layer {
 public:
  explicit RoPELayer(DeviceType device_type, int32_t dim, int32_t kv_dim, int32_t head_size);

  Status check() const override;

  Status forward() override;

 private:
  int32_t dim_ = 0;
  int32_t kv_dim_ = 0;
  int32_t head_size_ = 0;
};
}  
#endif  // MYVLLM_LAYERS_ROPE_H_
