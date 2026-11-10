#ifndef MYVLLM_UTILS_FP16_H_
#define MYVLLM_UTILS_FP16_H_

#include <cstdint>
#include <cstring>

namespace my_vllm
{
inline float fp16_to_float(uint16_t bits)
{
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fu;
    uint32_t mantissa = bits & 0x03ffu;
    uint32_t output;
    if (exponent == 0)
    {
        if (mantissa == 0) output = sign;
        else
        {
            int shift = 0;
            while ((mantissa & 0x0400u) == 0) { mantissa <<= 1; ++shift; }
            mantissa &= 0x03ffu;
            output = sign | (static_cast<uint32_t>(127 - 14 - shift) << 23) | (mantissa << 13);
        }
    }
    else if (exponent == 0x1fu)
        output = sign | 0x7f800000u | (mantissa << 13);
    else
        output = sign | ((exponent + (127 - 15)) << 23) | (mantissa << 13);
    float value;
    std::memcpy(&value, &output, sizeof(value));
    return value;
}

inline uint16_t float_to_fp16(float value)
{
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0)
    {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<uint16_t>(sign | half_mantissa);
    }
    if (exponent >= 31)
    {
        if (((bits >> 23) & 0xffu) == 0xffu && mantissa != 0)
            return static_cast<uint16_t>(sign | 0x7c00u | (mantissa >> 13) | 1u);
        return static_cast<uint16_t>(sign | 0x7c00u);
    }
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u)))
    {
        ++half_mantissa;
        if (half_mantissa == 0x400u)
        {
            const int32_t rounded_exponent = exponent + 1;
            if (rounded_exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | (rounded_exponent << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}
}

#endif
