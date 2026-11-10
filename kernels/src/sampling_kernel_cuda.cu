#include <cub/block/block_reduce.cuh>
#include <cfloat>
#include <cuda_fp16.h>
#include <curand_kernel.h>
#include <cstring>

#include "cuda_alloc.h"
#include "sampling_kernel_cuda.cuh"

namespace my_vllm
{
namespace
{
struct MaxCandidate
{
    float value;
    int32_t token;
};

struct MaxCandidateOp
{
    __device__ MaxCandidate operator()(const MaxCandidate& left,
                                       const MaxCandidate& right) const
    {
        if (right.value > left.value ||
            (right.value == left.value && right.token < left.token))
            return right;
        return left;
    }
};

__global__ void sample_gumbel_kernel(const __half* logits, int32_t vocab_size,
                                    int32_t* metadata)
{
    const int32_t row = blockIdx.x;
    const int32_t tid = threadIdx.x;
    int32_t* row_metadata = metadata + static_cast<size_t>(row) * 4;
    const float temperature = __int_as_float(row_metadata[0]);
    const uint64_t seed = static_cast<uint32_t>(row_metadata[1]) |
        (static_cast<uint64_t>(static_cast<uint32_t>(row_metadata[2])) << 32);
    const __half* row_logits = logits + static_cast<size_t>(row) * vocab_size;

    curandStatePhilox4_32_10_t random_state;
    if (temperature > 1.0e-6f) curand_init(seed, tid, 0, &random_state);

    MaxCandidate local{-FLT_MAX, INT32_MAX};
    for (int32_t token = tid; token < vocab_size; token += blockDim.x)
    {
        float score = __half2float(row_logits[token]);
        if (temperature > 1.0e-6f)
        {
            const float uniform = fminf(curand_uniform(&random_state), 1.0f - 1.0e-7f);
            score += temperature * -logf(-logf(uniform));
        }
        if (score > local.value || (score == local.value && token < local.token))
            local = {score, token};
    }

    using BlockReduce = cub::BlockReduce<MaxCandidate, 256>;
    __shared__ typename BlockReduce::TempStorage storage;
    const MaxCandidate best = BlockReduce(storage).Reduce(local, MaxCandidateOp{});
    if (tid == 0) row_metadata[3] = best.token;
}
}

void sample_logits_gumbel_cuda(const Tensor& logits,
                               const std::vector<float>& temperatures,
                               const std::vector<uint64_t>& seeds,
                               std::vector<int32_t>& token_ids,
                               CudaConfig* config)
{
    CHECK(config != nullptr && config->stream != nullptr);
    CHECK_EQ(temperatures.size(), seeds.size());
    CHECK_GT(temperatures.size(), 0u);
    CHECK_EQ(logits.dims_size(), 2);
    const int32_t rows = static_cast<int32_t>(temperatures.size());
    const int32_t vocab_size = logits.get_dim(1);
    CHECK_EQ(logits.get_dim(0), rows);

    std::vector<int32_t> host_metadata(static_cast<size_t>(rows) * 4, 0);
    for (int32_t row = 0; row < rows; ++row)
    {
        int32_t temperature_bits = 0;
        std::memcpy(&temperature_bits, &temperatures[static_cast<size_t>(row)],
                    sizeof(temperature_bits));
        const uint64_t seed = seeds[static_cast<size_t>(row)];
        host_metadata[static_cast<size_t>(row) * 4] = temperature_bits;
        host_metadata[static_cast<size_t>(row) * 4 + 1] = static_cast<int32_t>(seed);
        host_metadata[static_cast<size_t>(row) * 4 + 2] =
            static_cast<int32_t>(seed >> 32);
    }

    Tensor device_metadata(DataType::kDataTypeInt32, rows, 4, true,
                           CUDADeviceAllocatorFactory::get_instance());
    const size_t metadata_bytes = host_metadata.size() * sizeof(int32_t);
    cudaError_t status = cudaMemcpyAsync(device_metadata.ptr<int32_t>(),
        host_metadata.data(), metadata_bytes, cudaMemcpyHostToDevice, config->stream);
    CHECK_EQ(status, cudaSuccess);
    constexpr int32_t threads = 256;
    sample_gumbel_kernel<<<rows, threads, 0, config->stream>>>(
        reinterpret_cast<const __half*>(logits.ptr<uint16_t>()), vocab_size,
        device_metadata.ptr<int32_t>());
    CHECK_EQ(cudaGetLastError(), cudaSuccess);
    status = cudaMemcpyAsync(host_metadata.data(), device_metadata.ptr<int32_t>(),
        metadata_bytes, cudaMemcpyDeviceToHost, config->stream);
    CHECK_EQ(status, cudaSuccess);
    status = cudaStreamSynchronize(config->stream);
    CHECK_EQ(status, cudaSuccess);

    token_ids.resize(static_cast<size_t>(rows));
    for (int32_t row = 0; row < rows; ++row)
        token_ids[static_cast<size_t>(row)] =
            host_metadata[static_cast<size_t>(row) * 4 + 3];
}
}
