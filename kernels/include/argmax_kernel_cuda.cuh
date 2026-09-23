#ifndef MYVLLM_KERNELS_ARGMAX_H_
#define MYVLLM_KERNELS_ARGMAX_H_
namespace my_vllm 
{
size_t argmax_kernel_cu(const float* input_ptr, size_t size, void* stream);
}
#endif  // MYVLLM_KERNELS_ARGMAX_H_
