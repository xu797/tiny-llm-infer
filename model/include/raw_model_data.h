#ifndef MYVLLM_MODEL_RAW_MODEL_DATA_H_
#define MYVLLM_MODEL_RAW_MODEL_DATA_H_

#include <cstddef>
#include <cstdint>

namespace my_vllm 
{
struct RawModelData 
{
    // ~RawModelData();
    virtual ~RawModelData();

    int32_t fd = -1;
    size_t file_size = 0; 
    void* data = nullptr;
    void* weight_data = nullptr;

    virtual const void* weight(size_t offset) const = 0;
};

struct RawModelDataFp32 : RawModelData
{
    const void* weight(size_t offset) const override;
};

struct RawModelDataInt8 : RawModelData 
{
    const void* weight(size_t offset) const override;
};

}  
#endif  
