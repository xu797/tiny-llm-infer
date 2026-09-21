#ifndef MYVLLM_LAYERS_LAYER_H_
#define MYVLLM_LAYERS_LAYER_H_

#include <string>
#include <vector>

#include "status.h"
#include "tensor.h"
#include "cuda_config.h"

namespace my_vllm
{

enum class LayerType : uint8_t 
{
    kLayerUnknown = 0,
    kLayerLinear = 1,
    kLayerEncode = 2,
    kLayerEmbedding = 3,
    kLayerRMSNorm = 4,
    kLayerMatmul = 5,
    kLayerRoPe = 6,
    kLayerMHA = 7,
    kLayerSoftmax = 8,
    kLayerAdd = 9,
    kLayerSwiGLU = 10,
};

class BaseLayer 
{
public:
    explicit BaseLayer(DeviceType device_type, LayerType layer_type, DataType data_type, std::string layer_name = "");

    DataType data_type() const;

    LayerType layer_type() const;

    virtual Status init() = 0;

    virtual Status forward() = 0;

    virtual Status forward(const Tensor& input1, const Tensor& output1) = 0;

    virtual Status forward(const Tensor& input1, const Tensor& input2,
                                const Tensor& output1) = 0;

    virtual Status forward(const Tensor& input1, const Tensor& input2,
                                const Tensor& input3, const Tensor& output1) = 0;

    virtual Status forward(const Tensor& input1, const Tensor& input2,
                                const Tensor& input3, const Tensor& input4,
                                const Tensor& output1) = 0;

    virtual Status forward(const Tensor& input1, const Tensor& input2,
                                const Tensor& input3, const Tensor& input4,
                                const Tensor& input5, const Tensor& output1) = 0;

    virtual void set_input(int32_t idx, const Tensor& input) = 0;

    virtual void set_output(int32_t idx, const Tensor& output) = 0;

    virtual size_t input_size() const = 0;

    virtual size_t output_size() const = 0;

    virtual Status check() const = 0;

    virtual Tensor& get_input(int32_t idx) = 0;

    virtual Tensor& get_output(int32_t idx) = 0;

    virtual const Tensor& get_input(int32_t idx) const = 0;

    virtual const Tensor& get_output(int32_t idx) const = 0;

    virtual Status set_weight(int32_t idx, const Tensor& weight);

    virtual Status set_weight(int32_t idx, const std::vector<int32_t>& dims,
                                    const void* weight_ptr,
                                    DeviceType device_type = DeviceType::kDeviceUnknown);

    const std::string& get_layer_name() const;

    void set_layer_name(const std::string& layer_name);

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

protected:
    std::string layer_name_;
    LayerType layer_type_ = LayerType::kLayerUnknown;
    DataType data_type_ = DataType::kDataTypeUnknown;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

class Layer : public BaseLayer
{
public:
    explicit Layer(DeviceType device_type, LayerType layer_type, std::string layer_name = "");

    Status init() override;

    Status check_tensor(const Tensor& tensor, DeviceType device_type,
                            DataType data_type) const;

    Status check_tensor_with_dim(const Tensor& tensor, DeviceType device_type,
                                        DataType data_type, ...) const;

    Status check() const override;

    Status forward() override;

    Status forward(const Tensor& input1, const Tensor& output1) override;

    Status forward(const Tensor& input1, const Tensor& input2,
                        const Tensor& output1) override;

    Status forward(const Tensor& input1, const Tensor& input2,
                        const Tensor& input3, const Tensor& output1) override;

    Status forward(const Tensor& input1, const Tensor& input2,
                        const Tensor& input3, const Tensor& input4,
                        const Tensor& output1) override;

    Status forward(const Tensor& input1, const Tensor& input2,
                        const Tensor& input3, const Tensor& input4,
                        const Tensor& input5, const Tensor& output1) override;

    void set_input(int32_t idx, const Tensor& input) override;

    void set_output(int32_t idx, const Tensor& output) override;

    const Tensor& get_input(int32_t idx) const override;

    const Tensor& get_output(int32_t idx) const override;

    Tensor& get_input(int32_t idx) override;

    Tensor& get_output(int32_t idx) override;

    size_t input_size() const override;

    size_t output_size() const override;

    void reset_input_size(size_t size);

    void reset_output_size(size_t size);

    virtual void to_cuda();

    void set_cuda_config(std::shared_ptr<CudaConfig> config);

    std::shared_ptr<CudaConfig> cuda_config() const;

protected:
    std::vector<Tensor> inputs_;
    std::vector<Tensor> outputs_;
    std::shared_ptr<CudaConfig> cuda_config_;
};

class LayerParam : public Layer 
{
public:
    explicit LayerParam(DeviceType device_type, LayerType layer_type,
                        bool is_quant_layer = false, std::string layer_name = "");

    size_t weight_size() const;

    void reset_weight_size(size_t size);

    Tensor& get_weight(int32_t idx);

    const Tensor& get_weight(int32_t idx) const;

    void to_cuda() override;

    Status set_weight(int32_t idx, const Tensor& weight) override;

    Status set_weight(int32_t idx, const std::vector<int32_t>& dims, const void* weight_ptr,
                            DeviceType device_type = DeviceType::kDeviceUnknown) override;

    void set_scales(const Tensor& scales);

    void set_group_size(int32_t group_size);

    int32_t get_scale_num() const;

protected:
    int32_t group_size_ = 0;
    bool is_quant_layer_ = false;
    Tensor scales_;
    std::vector<Tensor> weights_;
};

}

#endif