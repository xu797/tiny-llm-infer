#include "add_kernel_cpu.h"
#include "fp16.h"

namespace my_vllm
{

void add_kernel_cpu(const Tensor& input1, const Tensor& input2, const Tensor& output, void* stream)
{
    UNUSED(stream);
    CHECK_EQ(input1.is_empty(), false);
    CHECK_EQ(input2.is_empty(), false);
    CHECK_EQ(output.is_empty(), false);

    CHECK_EQ(input1.size(), input2.size());
    CHECK_EQ(input1.size(), output.size());

    const uint16_t* a = input1.ptr<uint16_t>();
    const uint16_t* b = input2.ptr<uint16_t>();
    uint16_t* y = const_cast<uint16_t*>(output.ptr<uint16_t>());
    for (size_t index = 0; index < input1.size(); ++index)
        y[index] = float_to_fp16(fp16_to_float(a[index]) + fp16_to_float(b[index]));
}
}
