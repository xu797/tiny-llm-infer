#ifndef MYVLLM_KERNELS_SAMPLING_CUDA_H_
#define MYVLLM_KERNELS_SAMPLING_CUDA_H_

#include <cstdint>
#include <vector>

#include "cuda_config.h"
#include "tensor.h"

namespace my_vllm
{
void sample_logits_gumbel_cuda(const Tensor& logits,
                               const std::vector<float>& temperatures,
                               const std::vector<uint64_t>& seeds,
                               std::vector<int32_t>& token_ids,
                               CudaConfig* config);
}

#endif
