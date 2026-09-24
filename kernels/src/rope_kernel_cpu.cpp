#include "rope_kernel_cpu.h"

namespace my_vllm
{
#if defined (LLAMA3_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(500000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                     const Tensor& input_k, const Tensor& input_pos,
                     const Tensor& sin_cache, const Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim ++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec =
            const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                      : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#elif defined (QWEN2_SUPPORT)
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(1000000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                     const Tensor& input_k, const Tensor& input_pos,
                     const Tensor& sin_cache, const Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += head_size) {
    for (int32_t head_dim = i % head_size; head_dim < head_size / 2; head_dim ++) {
      float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim * 2);
      float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim * 2);

      int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
      for (int32_t v = 0; v < rotn; v++) {
        float* vec =
            const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                      : input_k.ptr<float>());  // the vector to rotate (query or key)
        float v0 = vec[i + head_dim];
        float v1 = vec[i + head_dim + head_size / 2];
        vec[i + head_dim] = v0 * fcr - v1 * fci;
        vec[i + head_dim + head_size / 2] = v0 * fci + v1 * fcr;
      }
    }
  }
}
#else
void sin_cos_cache_calc_cpu(int head_size, int max_seq_len, float* sin_cache, float* cos_cache) {
  for (int pos = 0; pos < max_seq_len; ++pos) {
    for (int head_dim = 0; head_dim < head_size; ++head_dim) {
      float freq =
          1.0f / std::pow(10000.0f, static_cast<float>(head_dim) / static_cast<float>(head_size));
      float val = static_cast<float>(pos) * freq;
      float fcr = cosf(val);
      float fci = sinf(val);
      *(sin_cache + pos * head_size + head_dim) = fci;
      *(cos_cache + pos * head_size + head_dim) = fcr;
    }
  }
}

void rope_kernel_cpu(int32_t dim, int32_t kv_dim, int32_t head_size, const Tensor& input_q,
                     const Tensor& input_k, const Tensor& input_pos,
                     const Tensor& sin_cache, const Tensor& cos_cache,
                     void* stream) {
  UNUSED(stream);
  const int32_t pos = *input_pos.ptr<int32_t>(0);

  for (int32_t i = 0; i < dim; i += 2) {
    int32_t head_dim = i % head_size;
    float fci = *(sin_cache.ptr<float>() + pos * head_size + head_dim);
    float fcr = *(cos_cache.ptr<float>() + pos * head_size + head_dim);

    int32_t rotn = i < kv_dim ? 2 : 1;  // how many vectors? 2 = q & k, 1 = q only
    for (int32_t v = 0; v < rotn; v++) {
      float* vec =
          const_cast<float*>(v == 0 ? input_q.ptr<float>()
                                    : input_k.ptr<float>());  // the vector to rotate (query or key)
      float v0 = vec[i];
      float v1 = vec[i + 1];
      vec[i] = v0 * fcr - v1 * fci;
      vec[i + 1] = v0 * fci + v1 * fcr;
    }
  }
}
#endif

void qwen3_sin_cos_cache_calc_cpu(int32_t head_size, int32_t max_seq_len, float rope_theta,
                                  float* sin_cache, float* cos_cache)
{
  const int32_t half = head_size / 2;
  for (int32_t pos = 0; pos < max_seq_len; ++pos)
  {
    for (int32_t i = 0; i < half; ++i)
    {
      const float frequency = std::pow(rope_theta, -2.0f * static_cast<float>(i) / head_size);
      const float angle = static_cast<float>(pos) * frequency;
      sin_cache[pos * head_size + i] = std::sin(angle);
      cos_cache[pos * head_size + i] = std::cos(angle);
    }
  }
}

void qwen3_rope_kernel_cpu(int32_t query_heads, int32_t kv_heads, int32_t head_size,
                           const Tensor& input_q, const Tensor& input_k,
                           const Tensor& input_pos, const Tensor& sin_cache,
                           const Tensor& cos_cache, void* stream)
{
  UNUSED(stream);
  const int32_t pos = input_pos.ptr<int32_t>()[0];
  const float* sin_row = sin_cache.ptr<float>() + pos * head_size;
  const float* cos_row = cos_cache.ptr<float>() + pos * head_size;
  float* query = const_cast<float*>(input_q.ptr<float>());
  float* key = const_cast<float*>(input_k.ptr<float>());
  const int32_t half = head_size / 2;

  for (int32_t head = 0; head < query_heads; ++head)
  {
    for (int32_t i = 0; i < half; ++i)
    {
      const int32_t first = head * head_size + i;
      const int32_t second = first + half;
      const float x = query[first];
      const float y = query[second];
      query[first] = x * cos_row[i] - y * sin_row[i];
      query[second] = x * sin_row[i] + y * cos_row[i];
    }
  }
  for (int32_t head = 0; head < kv_heads; ++head)
  {
    for (int32_t i = 0; i < half; ++i)
    {
      const int32_t first = head * head_size + i;
      const int32_t second = first + half;
      const float x = key[first];
      const float y = key[second];
      key[first] = x * cos_row[i] - y * sin_row[i];
      key[second] = x * sin_row[i] + y * cos_row[i];
    }
  }
}
}  // namespace my_vllm
