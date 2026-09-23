#include "scale_kernel_cpu.h"

namespace my_vllm 
{
void scale_inplace_cpu(float scale, const Tensor& tensor, void* stream)
{
    UNUSED(stream);
    CHECK(tensor.is_empty() == false);
    arma::fvec tensor_mat(const_cast<float*>(tensor.ptr<float>()), tensor.size(), false, true);
    tensor_mat = tensor_mat * scale;
}
}  