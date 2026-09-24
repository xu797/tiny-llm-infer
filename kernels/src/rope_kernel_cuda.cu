#include "rope_kernel_cuda.cuh"

namespace my_vllm 
{

#if defined (LLAMA3_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#elif defined (QWEN2_SUPPORT)
__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;

  int num_heads = dim / head_size;
  int head_pair_count = head_size / 2;
  int total_pairs = num_heads * head_pair_count;
  if (idx > total_pairs) {
    return;
  }

  int head_idx = idx / head_pair_count;
  int head_dim = idx % head_pair_count;

  int i = head_idx * head_size;
  int v0_idx = i + head_dim;
  int v1_idx = i + head_dim + head_size / 2;

  float fci = sin_cache[pos * head_size + head_dim * 2];
  float fcr = cos_cache[pos * head_size + head_dim * 2];

  int rotn = i < kv_dim ? 2 : 1;

  for (int v = 0; v < rotn; v++) {
    float* vec = const_cast<float*>(v == 0 ? input_q : input_k);  // the vector to rotate (query or key)
    float v0 = vec[v0_idx];
    float v1 = vec[v1_idx];
    vec[v0_idx] = fcr * v0 - fci * v1;
    vec[v1_idx] = fcr * v1 + fci * v0;
  }
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  int head_dim = idx % head_size;
  for (int pos = 0; pos < max_seq_len; ++pos) {
    float freq = 1.0f / pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
    float val = static_cast<float>(pos) * freq;
    float fcr = cosf(val);
    float fci = sinf(val);
    *(sin_cache + pos * head_size + head_dim) = fci;
    *(cos_cache + pos * head_size + head_dim) = fcr;
  }
}
#else
__device__ void rope_calc(float fcr, float fci, float* vec, int32_t idx) 
{
    float2* vec_ptr = reinterpret_cast<float2*>(vec + idx);
    float2 vec_value = *vec_ptr;
    *vec_ptr = make_float2(vec_value.x * fcr - vec_value.y * fci, vec_value.x * fci + vec_value.y * fcr);
}

__global__ void rope_kernel_cu_fp32(int pos, int dim, int kv_dim, int head_size,
                                    const float* input_q, const float* input_k,
                                    const float* sin_cache, const float* cos_cache) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    idx = idx * 2;
    if (idx >= dim)
    {
        return;
    }

    int head_dim = idx % head_size;
    float fci = *(sin_cache + pos * head_size + head_dim);
    float fcr = *(cos_cache + pos * head_size + head_dim);

    rope_calc(fcr, fci, const_cast<float*>(input_q), idx);
    if (idx >= kv_dim) 
    {
        return;
    }
    rope_calc(fcr, fci, const_cast<float*>(input_k), idx);
}

__global__ void sin_cos_calc(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) 
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int head_dim = idx % head_size;
    for (int pos = 0; pos < max_seq_len; ++pos) 
    {
        float freq = 1.0f / pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
        float val = static_cast<float>(pos) * freq;
        float fcr = cosf(val);
        float fci = sinf(val);
        *(sin_cache + pos * head_size + head_dim) = fci;
        *(cos_cache + pos * head_size + head_dim) = fcr;
    }
}
#endif

void sin_cos_cache_calc_cu(int head_size, int max_seq_len, const Tensor& sin_cache, const Tensor& cos_cache, cudaStream_t stream) 
{
    CHECK_EQ(sin_cache.is_empty(), false);
    CHECK_EQ(cos_cache.is_empty(), false);
    int threads = head_size;
    if (stream) 
    {
        sin_cos_calc<<<1, threads, 0, stream>>>(head_size, max_seq_len,
                                                const_cast<float*>(sin_cache.ptr<float>()),
                                                const_cast<float*>(cos_cache.ptr<float>()));
    } 
    else 
    {
        sin_cos_calc<<<1, threads>>>(head_size, max_seq_len, const_cast<float*>(sin_cache.ptr<float>()),
                                        const_cast<float*>(cos_cache.ptr<float>()));
    }
}

void rope_kernel_cu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                    const Tensor& input_k, const Tensor& input_pos,
                    const Tensor& sin_cache, const Tensor& cos_cache, void* stream) 
{
    const int32_t pos = *input_pos.ptr<int32_t>(0);
    int threads = 128;
    int blocks = (dim + threads - 1) / threads;
    if (stream) 
    {
        cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
        rope_kernel_cu_fp32<<<blocks, threads, 0, stream_>>>(
            pos, dim, kv_dim, head_size, input_q.ptr<float>(), input_k.ptr<float>(),
            sin_cache.ptr<float>(), cos_cache.ptr<float>());
    } 
    else 
    {
        rope_kernel_cu_fp32<<<blocks, threads>>>(pos, dim, kv_dim, head_size, input_q.ptr<float>(),
                                                    input_k.ptr<float>(), sin_cache.ptr<float>(),
                                                    cos_cache.ptr<float>());
    }
    }

__global__ void qwen3_sin_cos_calc(int32_t head_size, int32_t max_seq_len, float rope_theta,
                                   float* sin_cache, float* cos_cache)
{
    const int32_t i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i >= head_size / 2) return;
    const float frequency = powf(rope_theta, -2.0f * static_cast<float>(i) / head_size);
    for (int32_t pos = 0; pos < max_seq_len; ++pos)
    {
        const float angle = static_cast<float>(pos) * frequency;
        sin_cache[pos * head_size + i] = sinf(angle);
        cos_cache[pos * head_size + i] = cosf(angle);
    }
}

__global__ void qwen3_rope_kernel(int32_t pos, int32_t query_heads, int32_t kv_heads,
                                  int32_t head_size, float* query, float* key,
                                  const float* sin_cache, const float* cos_cache)
{
    const int32_t half = head_size / 2;
    const int32_t pair = threadIdx.x + blockDim.x * blockIdx.x;
    const int32_t total_pairs = query_heads * half;
    if (pair >= total_pairs) return;
    const int32_t head = pair / half;
    const int32_t i = pair % half;
    const int32_t first = head * head_size + i;
    const int32_t second = first + half;
    const float sine = sin_cache[pos * head_size + i];
    const float cosine = cos_cache[pos * head_size + i];

    const float qx = query[first];
    const float qy = query[second];
    query[first] = qx * cosine - qy * sine;
    query[second] = qx * sine + qy * cosine;
    if (head < kv_heads)
    {
        const float kx = key[first];
        const float ky = key[second];
        key[first] = kx * cosine - ky * sine;
        key[second] = kx * sine + ky * cosine;
    }
}

void qwen3_sin_cos_cache_calc_cu(int32_t head_size, int32_t max_seq_len, float rope_theta,
                                 const Tensor& sin_cache, const Tensor& cos_cache,
                                 cudaStream_t stream)
{
    CHECK(!sin_cache.is_empty() && !cos_cache.is_empty());
    const int32_t threads = 128;
    const int32_t blocks = (head_size / 2 + threads - 1) / threads;
    qwen3_sin_cos_calc<<<blocks, threads, 0, stream>>>(
        head_size, max_seq_len, rope_theta, const_cast<float*>(sin_cache.ptr<float>()),
        const_cast<float*>(cos_cache.ptr<float>()));
}

void qwen3_rope_kernel_cu(int32_t query_heads, int32_t kv_heads, int32_t head_size,
                          const Tensor& input_q, const Tensor& input_k,
                          const Tensor& input_pos, const Tensor& sin_cache,
                          const Tensor& cos_cache, void* stream)
{
    const int32_t pos = input_pos.ptr<int32_t>()[0];
    const int32_t total_pairs = query_heads * (head_size / 2);
    const int32_t threads = 128;
    const int32_t blocks = (total_pairs + threads - 1) / threads;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    qwen3_rope_kernel<<<blocks, threads, 0, cuda_stream>>>(
        pos, query_heads, kv_heads, head_size, const_cast<float*>(input_q.ptr<float>()),
        const_cast<float*>(input_k.ptr<float>()), sin_cache.ptr<float>(), cos_cache.ptr<float>());
}
}
