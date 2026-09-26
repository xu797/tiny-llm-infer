#ifndef MYVLLM_ENGINE_SAMPLING_PARAMS_H_
#define MYVLLM_ENGINE_SAMPLING_PARAMS_H_

#include <cstdint>

#include "status.h"

namespace my_vllm::engine
{
struct SamplingParams
{
    int32_t max_tokens = 64;
    float temperature = 0.0f;
    float top_p = 1.0f;
    bool ignore_eos = false;
    uint64_t seed = 0;

    Status validate() const
    {
        if (max_tokens < 0)
        {
            return InvalidArgument("max_tokens cannot be negative.");
        }
        if (temperature < 0.0f)
        {
            return InvalidArgument("temperature cannot be negative.");
        }
        if (top_p <= 0.0f or top_p > 1.0f)
        {
            return InvalidArgument("top_p must be in the interval (0, 1].");
        }
        return Success();
    }
};
}

#endif
