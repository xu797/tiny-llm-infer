#include "swiglu_kernel_cpu.h"
#include "fp16.h"

namespace my_vllm 
{
void swiglu_kernel_cpu(const Tensor& input1, const Tensor& input2, const Tensor& output, void* stream) 
{
    UNUSED(stream);
    CHECK_EQ(input1.is_empty(),false);
    CHECK_EQ(input2.is_empty(),false);
    CHECK_EQ(output.is_empty(),false);

    CHECK(input1.device_type() == DeviceType::kDeviceCPU);
    CHECK(input2.device_type() == DeviceType::kDeviceCPU);
    CHECK(output.device_type() == DeviceType::kDeviceCPU);

    const uint16_t* gate = input1.ptr<uint16_t>();
    const uint16_t* up = input2.ptr<uint16_t>();
    uint16_t* out = const_cast<uint16_t*>(output.ptr<uint16_t>());
    for (size_t i = 0; i < input1.size(); ++i)
    {
        const float x = fp16_to_float(gate[i]);
        out[i] = float_to_fp16((x / (1.0f + std::exp(-x))) * fp16_to_float(up[i]));
    }
    }
}  // namespace my_vllm
