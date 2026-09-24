
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <sentencepiece_processor.h>
#include <utility>

#include "llama.h"
#include "matmul.h"
#include "mha.h"
#include "rmsnorm.h"
#include "rope_kernel_cpu.h"
#include "rope_kernel_cuda.cuh"
#include "tick.h"
#include "cpu_alloc.h"
#include "cuda_alloc.h"

namespace my_vllm
{

    void LLama2Layers::to_cuda(std::shared_ptr<CudaConfig> config)
    {
        if (add_layer_)
        {
            add_layer_->set_cuda_config(config);
            add_layer_->to_cuda();
        }

        if (rope_layer_)
        {
            rope_layer_->set_cuda_config(config);
            rope_layer_->to_cuda();
        }

        if (swiglu_layer_)
        {
            swiglu_layer_->set_cuda_config(config);
            swiglu_layer_->to_cuda();
        }

        if (cls_layer_)
        {
            cls_layer_->set_cuda_config(config);
            cls_layer_->to_cuda();
        }

        if (embedding_layer_)
        {
            embedding_layer_->set_cuda_config(config);
            embedding_layer_->to_cuda();
        }

        if (mha_layer_)
        {
            mha_layer_->set_cuda_config(config);
            mha_layer_->to_cuda();
        }

        for (auto &weight_layer : wq_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : wk_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : wv_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : wo_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : w1_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : w2_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &weight_layer : w3_layers_)
        {
            if (weight_layer)
            {
                weight_layer->set_cuda_config(config);
                weight_layer->to_cuda();
            }
        }

        for (auto &rms_norm_layer : rmsnorm_layers_)
        {
            if (rms_norm_layer)
            {
                rms_norm_layer->set_cuda_config(config);
                rms_norm_layer->to_cuda();
            }
        }
    }

    LLama2Model::LLama2Model(TokenizerType tokenizer_type, std::string token_path, std::string model_path, bool is_quant_model)
        : Model(tokenizer_type, ModelType::kModelTypeLLama2, std::move(token_path), std::move(model_path), is_quant_model)
    {
    }

    LLama2Model::~LLama2Model()
    {
        if (cuda_config_ && cuda_config_->stream)
        {
            cudaStreamSynchronize(cuda_config_->stream);
            cudaStreamDestroy(cuda_config_->stream);
            cuda_config_->stream = nullptr;
        }
    }

    Status LLama2Model::init(DeviceType device_type)
    {
        if (token_path_.empty())
        {
            return PathNotValid(token_path_);
        }
        if (device_type == DeviceType::kDeviceCPU && is_quant_model_)
        {
            return InternalError("The cpu device do not support int8 quant model.");
        }
        if (device_type != DeviceType::kDeviceCPU && device_type != DeviceType::kDeviceCUDA)
        {
            return InvalidArgument("The requested device type is not supported.");
        }

        device_type_ = device_type;
        if (device_type == DeviceType::kDeviceCUDA)
        {
            cudaError_t err = cudaSetDevice(0);
            if (err != cudaSuccess)
            {
                return InternalError(std::string("Failed to select CUDA device 0: ") + cudaGetErrorString(err));
            }
            cuda_config_ = std::make_shared<CudaConfig>();
            err = cudaStreamCreate(&cuda_config_->stream);
            if (err != cudaSuccess)
            {
                return InternalError(std::string("Failed to create CUDA stream: ") + cudaGetErrorString(err));
            }
        }

        Status read_status = gen_model_from_file();
        if (!read_status)
        {
            return read_status;
        }
        init_mem();
        if (device_type_ == DeviceType::kDeviceCPU)
        {
            sin_cos_cache_calc_cpu(config_->head_size_, config_->seq_len_,
                                   get_buffer(ModelBufferType::kSinCache).ptr<float>(),
                                   get_buffer(ModelBufferType::kCosCache).ptr<float>());
        }
        else
        {
            CHECK_NE(cuda_config_, nullptr);
            sin_cos_cache_calc_cu(config_->head_size_, config_->seq_len_,
                                  get_buffer(ModelBufferType::kSinCache),
                                  get_buffer(ModelBufferType::kCosCache), cuda_config_->stream);
        }

        sampler_ = std::make_unique<ArgmaxSampler>(device_type_);
        return Success();
    }

    Status LLama2Model::forward(const Tensor &input, const Tensor &pos_tensor, int &next) const
    {
        if (input.is_empty())
        {
            return InvalidArgument("The input tensor is empty.");
        }
        if (!config_ || input.data_type() != DataType::kDataTypeFp32 ||
            input.device_type() != device_type_ || input.size() != static_cast<size_t>(config_->dim_))
        {
            return InvalidArgument("The model input must be one fp32 hidden-state vector on the model device.");
        }
        if (pos_tensor.is_empty() || pos_tensor.data_type() != DataType::kDataTypeInt32 ||
            pos_tensor.device_type() != DeviceType::kDeviceCPU || pos_tensor.size() != 1)
        {
            return InvalidArgument("The position tensor must contain one CPU int32 value.");
        }
        const int32_t pos = pos_tensor.ptr<int32_t>()[0];
        if (pos < 0 || pos >= config_->seq_len_)
        {
            return InvalidArgument("The input position is outside the model context window.");
        }
        if (device_type_ == DeviceType::kDeviceCPU && is_quant_model_)
        {
            return InternalError("Unsupported int8 quant in the cpu device");
        }

        for (int32_t layer_idx = 0; layer_idx < config_->layer_num_; ++layer_idx)
        {
            attention_rms(layer_idx, input);
            // attention (wq wk wv @ input)
            attention_qkv(layer_idx, pos_tensor);
            // multi-head attention
            attention_mha(layer_idx, pos_tensor);
            // feed forward
            feed_forward(layer_idx, input);
        }
        cls_logits(input);
        return Success();
    }

    void LLama2Model::create_nonparam_layers()
    {
        CHECK(llama_layers_ != nullptr);
        llama_layers_->rope_layer_ = std::make_shared<RoPELayer>(
            device_type_, config_->dim_, config_->kv_dim_, config_->head_size_);

        llama_layers_->mha_layer_ = std::make_shared<MultiHeadAttention>(
            device_type_, 0, config_->kv_mul_, config_->kv_dim_, config_->seq_len_, config_->head_num_,
            config_->head_size_);

        llama_layers_->add_layer_ = std::make_shared<VecAddLayer>(device_type_);

        llama_layers_->swiglu_layer_ =
            std::make_shared<SwiGLULayer>(device_type_, config_->hidden_dim_);
    }

    void LLama2Model::create_param_quant_layers()
    {
        CHECK(is_quant_model_);
        CHECK(llama_layers_ != nullptr);

        size_t pos = 0;
        int32_t dim = config_->dim_;
        auto cpu_device_type = DeviceType::kDeviceCPU;

        // query
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wq = std::make_shared<MatmulLayer>(device_type_, dim, dim, true);
            wq->set_group_size(group_size_);
            wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wq_layers_.push_back(wq);
            pos = pos + dim * dim + wq->get_scale_num() * sizeof(float);
        }

        // key
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wk = std::make_shared<MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
            wk->set_group_size(group_size_);
            wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wk_layers_.push_back(wk);
            pos = pos + config_->kv_dim_ * dim + wk->get_scale_num() * sizeof(float);
        }

        // value
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wv = std::make_shared<MatmulLayer>(device_type_, config_->kv_dim_, dim, true);
            wv->set_group_size(group_size_);
            wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wv_layers_.push_back(wv);
            pos += config_->kv_dim_ * dim + wv->get_scale_num() * sizeof(float);
        }

        // output
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wo = std::make_shared<MatmulLayer>(device_type_, dim, dim, true);
            wo->set_group_size(group_size_);
            wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wo_layers_.push_back(wo);
            pos = pos + dim * dim + wo->get_scale_num() * sizeof(float);
        }

        // w1 layers
        int32_t hidden_dim = config_->hidden_dim_;
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w1 = std::make_shared<MatmulLayer>(device_type_, hidden_dim, dim, true);
            w1->set_group_size(group_size_);
            w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w1_layers_.push_back(w1);
            pos = pos + dim * hidden_dim + w1->get_scale_num() * sizeof(float);
        }

        // w2 layers
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w2 = std::make_shared<MatmulLayer>(device_type_, dim, hidden_dim, true);
            w2->set_group_size(group_size_);
            w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w2_layers_.push_back(w2);
            pos = pos + dim * hidden_dim + w2->get_scale_num() * sizeof(float);
        }

        // w3 layers
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w3 = std::make_shared<MatmulLayer>(device_type_, hidden_dim, dim, true);
            w3->set_group_size(group_size_);
            w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w3_layers_.push_back(w3);
            pos = pos + dim * hidden_dim + w3->get_scale_num() * sizeof(float);
        }

        // wcls layer
        auto cls_layer = std::make_shared<MatmulLayer>(device_type_, config_->vocab_size_, dim, true);
        cls_layer->set_group_size(group_size_);
        if (config_->is_shared_weight_)
        {
            // using token embedding weight
            cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                                  cpu_device_type);
        }
        else
        {
            // no shared
            cls_layer->set_weight(0, {config_->vocab_size_, dim}, this->raw_model_data_->weight(pos),
                                  cpu_device_type);
            pos = pos + config_->vocab_size_ * dim + cls_layer->get_scale_num() * sizeof(float);
        }
        llama_layers_->cls_layer_ = cls_layer;

        // embedding layer
        float *weight_ptr = (float *)raw_model_data_->weight(pos);
        llama_layers_->embedding_layer_ = std::make_shared<EmbeddingLayer>(
            device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));
        llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), dim}, weight_ptr,
                                                    cpu_device_type);
        weight_ptr += config_->vocab_size_ * dim;

        // rmsnorm attention attention,ffn,final
        for (int32_t i = 0; i < 2 * config_->layer_num_ + 1; ++i)
        {
            std::shared_ptr<RmsNormLayer> rms_norm_layer =
                std::make_shared<RmsNormLayer>(device_type_, dim);

            rms_norm_layer->set_weight(0, {dim}, weight_ptr, cpu_device_type);
            llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
            weight_ptr += dim;
        }
    }

    void LLama2Model::create_param_layers()
    {
        CHECK(!is_quant_model_);
        CHECK(llama_layers_ != nullptr);
        // The embedding layer
        auto cpu_device_type = DeviceType::kDeviceCPU;
        llama_layers_->embedding_layer_ = std::make_shared<EmbeddingLayer>(device_type_, config_->dim_, config_->seq_len_, std::abs(config_->vocab_size_));

        const void *weight_embedding = raw_model_data_->weight(0);
        llama_layers_->embedding_layer_->set_weight(0, {std::abs(config_->vocab_size_), config_->dim_},
                                                    weight_embedding, cpu_device_type);

        // create all matmul layer
        int32_t dim = config_->dim_; // d_model
        size_t pos = dim * std::abs(config_->vocab_size_) + dim * config_->layer_num_;
        // create weight matrix for query
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wq = std::make_shared<MatmulLayer>(device_type_, dim, dim);
            wq->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wq_layers_.push_back(wq);
            pos += dim * dim;
        }

        // create weight matrix for key
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wk = std::make_shared<MatmulLayer>(device_type_, config_->kv_dim_, dim);
            wk->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wk_layers_.push_back(wk);
            pos += config_->kv_dim_ * dim;
        }

        // create weight matrix for value
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wv = std::make_shared<MatmulLayer>(device_type_, config_->kv_dim_, dim);
            wv->set_weight(0, {config_->kv_dim_, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wv_layers_.push_back(wv);
            pos += config_->kv_dim_ * dim;
        }

        // create weight matrix for output
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto wo = std::make_shared<MatmulLayer>(device_type_, dim, dim);
            wo->set_weight(0, {dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->wo_layers_.push_back(wo);
            pos += dim * dim;
        }

        // skip ffn rmsnorm
        pos += config_->layer_num_ * dim;

        // w1 layers
        int32_t hidden_dim = config_->hidden_dim_;
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w1 = std::make_shared<MatmulLayer>(device_type_, hidden_dim, dim);
            w1->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w1_layers_.push_back(w1);
            pos += dim * hidden_dim;
        }

        // w2 layers
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w2 = std::make_shared<MatmulLayer>(device_type_, dim, hidden_dim);
            w2->set_weight(0, {dim, hidden_dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w2_layers_.push_back(w2);
            pos += dim * hidden_dim;
        }

        // w3 layers
        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            auto w3 = std::make_shared<MatmulLayer>(device_type_, hidden_dim, dim);
            w3->set_weight(0, {hidden_dim, dim}, this->raw_model_data_->weight(pos), cpu_device_type);
            llama_layers_->w3_layers_.push_back(w3);
            pos += dim * hidden_dim;
        }

        // skip final rms weight
        pos += dim;
        // skip freqs_cos and freqs_sin weight
        pos += config_->seq_len_ * config_->head_size_;

        llama_layers_->cls_layer_ = std::make_shared<MatmulLayer>(device_type_, config_->vocab_size_, dim);
        if (config_->is_shared_weight_)
        {
            // using token embedding weight
            llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                                  this->raw_model_data_->weight(0), cpu_device_type);
        }
        else
        {
            llama_layers_->cls_layer_->set_weight(0, {config_->vocab_size_, dim},
                                                  this->raw_model_data_->weight(pos), cpu_device_type);
        }

        // create rmsnorm layer
        size_t rmsnorm_pos = config_->dim_ * std::abs(config_->vocab_size_);

        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            std::shared_ptr<RmsNormLayer> rms_norm_layer =
                std::make_shared<RmsNormLayer>(device_type_, config_->dim_);

            const void *weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
            rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
            llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);
            rmsnorm_pos += config_->dim_;
        }

        // skip attention.wq attention.wk attention.wv attention.wo
        rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;
        rmsnorm_pos +=
            config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
        rmsnorm_pos +=
            config_->layer_num_ * config_->dim_ * (config_->kv_head_num_ * config_->head_size_);
        rmsnorm_pos += config_->layer_num_ * config_->dim_ * config_->dim_;

        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            std::shared_ptr<RmsNormLayer> rms_norm_layer =
                std::make_shared<RmsNormLayer>(device_type_, config_->dim_);
            const void *weight_rmsnorm = raw_model_data_->weight(rmsnorm_pos);
            rms_norm_layer->set_weight(0, {config_->dim_}, weight_rmsnorm, cpu_device_type);
            llama_layers_->rmsnorm_layers_.push_back(rms_norm_layer);

            rmsnorm_pos += config_->dim_;
        }

        // skip ffn.w1 ffn.w2 ffn.w3
        rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
        rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;
        rmsnorm_pos += config_->layer_num_ * config_->hidden_dim_ * config_->dim_;

        std::shared_ptr<RmsNormLayer> rms_final_layer =
            std::make_shared<RmsNormLayer>(device_type_, config_->dim_);

        const void *weight_rmsnorm_final = raw_model_data_->weight(rmsnorm_pos);
        rms_final_layer->set_weight(0, {config_->dim_}, weight_rmsnorm_final, cpu_device_type);
        llama_layers_->rmsnorm_layers_.push_back(rms_final_layer);
    }

    std::vector<int32_t> LLama2Model::encode(const std::string &sentence) const
    {
        CHECK(encode_layer_ != nullptr);
        return encode_layer_->encode(sentence);
    }

    Status LLama2Model::generate(const std::string& prompt, int32_t max_new_tokens,
                                 std::string& output)
    {
        output.clear();
        if (!config_ || !encode_layer_ || !sampler_)
        {
            return InternalError("The model must be initialized before generation.");
        }
        if (max_new_tokens < 0)
        {
            return InvalidArgument("The number of generated tokens cannot be negative.");
        }
        if (max_new_tokens == 0)
        {
            return Success();
        }

        const std::vector<int32_t> token_ids = encode(prompt);
        if (token_ids.empty())
        {
            return InvalidArgument("The prompt did not produce any tokens.");
        }
        if (token_ids.size() > static_cast<size_t>(config_->seq_len_))
        {
            return InvalidArgument("The prompt is longer than the model context window.");
        }
        const int32_t max_context_tokens = config_->seq_len_ - static_cast<int32_t>(token_ids.size()) + 1;
        if (max_new_tokens > max_context_tokens)
        {
            return InvalidArgument("The requested generation length exceeds the model context window.");
        }

        const std::vector<int> prompt_tokens(token_ids.begin(), token_ids.end());
        const EmbeddingOutput prompt_embedding = embedding(prompt_tokens);
        Tensor& pos_tensor = get_buffer(ModelBufferType::kInputPos);
        int next = -1;
        for (size_t i = 0; i < token_ids.size(); ++i)
        {
            pos_tensor.index<int32_t>(0) = static_cast<int32_t>(i);
            const Tensor input = fill_input(pos_tensor, prompt_embedding, true);
            const bool skip_sampling = i + 1 < token_ids.size();
            const Status status = predict(input, pos_tensor, skip_sampling, next);
            if (!status)
            {
                return status;
            }
        }

        std::vector<int32_t> generated_tokens;
        generated_tokens.reserve(max_new_tokens);
        for (int32_t i = 0; i < max_new_tokens; ++i)
        {
            if (next < 0)
            {
                return InternalError("The model did not produce a token after processing the prompt.");
            }
            if (is_sentence_ending(next))
            {
                break;
            }
            generated_tokens.push_back(next);
            if (i + 1 == max_new_tokens)
            {
                break;
            }

            const std::vector<int> next_token{next};
            const EmbeddingOutput next_embedding = embedding(next_token);
            pos_tensor.index<int32_t>(0) = static_cast<int32_t>(token_ids.size()) + i;
            const Tensor input = fill_input(pos_tensor, next_embedding, false);
            const Status status = predict(input, pos_tensor, false, next);
            if (!status)
            {
                return status;
            }
        }
        output = decode(generated_tokens);
        return Success();
    }

    bool LLama2Model::is_sentence_ending(int32_t token_idx) const
    {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->is_sentence_ending(token_idx);
    }

    std::string LLama2Model::decode(int32_t token_idx) const
    {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->decode(token_idx);
    }

    std::string LLama2Model::decode(std::vector<int32_t> token_idxs) const
    {
        CHECK(this->encode_layer_ != nullptr);
        return this->encode_layer_->decode(token_idxs);
    }

    void LLama2Model::init_mem()
    {
        std::shared_ptr<DeviceAllocator> alloc;
        if (device_type_ == DeviceType::kDeviceCPU)
        {
            alloc = CPUDeviceAllocatorFactory::get_instance();
        }
        else
        {
            alloc = CUDADeviceAllocatorFactory::get_instance();
        }

        if (device_type_ == DeviceType::kDeviceCUDA)
        {
            CHECK_NE(cuda_config_, nullptr);
            llama_layers_->to_cuda(cuda_config_);
        }

        std::shared_ptr<DeviceAllocator> alloc_cpu = CPUDeviceAllocatorFactory::get_instance();
        std::shared_ptr<DeviceAllocator> alloc_cu = CUDADeviceAllocatorFactory::get_instance();

        Tensor input_tokens(DataType::kDataTypeInt32, 1, true, alloc_cpu);
        Tensor input_embeddings(DataType::kDataTypeFp32, 1, config_->dim_, true, alloc);
        Tensor sin_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);
        Tensor cos_cache(DataType::kDataTypeFp32, config_->head_size_ * config_->seq_len_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kSinCache, sin_cache));
        CHECK(insert_buffer(ModelBufferType::kCosCache, cos_cache));

        CHECK(insert_buffer(ModelBufferType::kInputTokens, input_tokens));
        CHECK(insert_buffer(ModelBufferType::kInputEmbeddings, input_embeddings));

        Tensor rms_output(DataType::kDataTypeFp32, config_->dim_, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kOutputRMSNorm, rms_output));
        CHECK(insert_buffer(ModelBufferType::kOutputMHA, rms_output));
        CHECK(insert_buffer(ModelBufferType::kW2Output, rms_output));
        CHECK(insert_buffer(ModelBufferType::kFFNRMSNorm, rms_output));

        Tensor w1_output(DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);
        Tensor w3_output(DataType::kDataTypeFp32, config_->hidden_dim_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kW1Output, w1_output));
        CHECK(insert_buffer(ModelBufferType::kW3Output, w3_output));

        // kv cache
        Tensor key_cache(DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                         config_->kv_dim_, true, alloc);
        Tensor value_cache(DataType::kDataTypeFp32, config_->layer_num_, config_->seq_len_,
                           config_->kv_dim_, true, alloc);

        CHECK(insert_buffer(ModelBufferType::kKeyCache, key_cache));
        CHECK(insert_buffer(ModelBufferType::kValueCache, value_cache));

        // Wq query output
        Tensor query(DataType::kDataTypeFp32, config_->dim_, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kQuery, query));

        // Pos tensor
        Tensor pos_tensor(DataType::kDataTypeInt32, 1, true, alloc_cpu);
        CHECK(insert_buffer(ModelBufferType::kInputPos, pos_tensor));

        // Attention output
        Tensor attn(DataType::kDataTypeFp32, config_->head_num_, config_->seq_len_, true, alloc);
        CHECK(insert_buffer(ModelBufferType::kScoreStorage, attn));
        CHECK(insert_buffer(ModelBufferType::kAttnOutput, query));

        // final forward output
        Tensor forward_output(DataType::kDataTypeFp32, config_->vocab_size_, true, alloc);
        if (device_type_ == DeviceType::kDeviceCUDA)
        {
            Tensor forward_output_cpu(DataType::kDataTypeFp32, config_->vocab_size_, true, alloc_cpu);
            CHECK(insert_buffer(ModelBufferType::kForwardOutputCPU, forward_output_cpu));
        }

        CHECK(insert_buffer(ModelBufferType::kForwardOutput, forward_output));
    }

    std::pair<Tensor, Tensor> LLama2Model::slice_kv_cache(int32_t layer_idx,
                                                          int32_t token_pos) const
    {
        int32_t layer_offset = layer_idx * config_->seq_len_ * config_->kv_dim_;
        int32_t cache_offset = layer_offset + token_pos * config_->kv_dim_;

        float *key_cache_ptr =
            const_cast<float *>(get_buffer(ModelBufferType::kKeyCache).ptr<float>(cache_offset));
        float *val_cache_ptr =
            const_cast<float *>(get_buffer(ModelBufferType::kValueCache).ptr<float>(cache_offset));

        auto key_cache = std::make_shared<Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                                  key_cache_ptr, true);
        auto val_cache = std::make_shared<Buffer>(config_->kv_dim_ * sizeof(float), nullptr,
                                                  val_cache_ptr, true);
        key_cache->set_device_type(device_type_);
        val_cache->set_device_type(device_type_);
        Tensor key(DataType::kDataTypeFp32, config_->kv_dim_);
        Tensor val(DataType::kDataTypeFp32, config_->kv_dim_);
        key.assign(key_cache);
        val.assign(val_cache);
        return {key, val};
    }

    Status LLama2Model::create_layers()
    {

        if (!llama_layers_)
        {
            llama_layers_ = std::make_unique<LLama2Layers>();
        }

        if (!is_quant_model_)
        {
            create_param_layers();
        }
        else
        {
            create_param_quant_layers();
        }
        create_nonparam_layers();

        if (!llama_layers_->embedding_layer_)
        {
            return InternalError("Create the embedding layer for the llama model failed!");
        }

        if (llama_layers_->rmsnorm_layers_.size() != 2 * config_->layer_num_ + 1)
        {
            return InternalError("Create the rmsnorm layers for the llama model failed!");
        }

        if (llama_layers_->wq_layers_.size() != config_->layer_num_ ||
            llama_layers_->wk_layers_.size() != config_->layer_num_ ||
            llama_layers_->wv_layers_.size() != config_->layer_num_ ||
            llama_layers_->wo_layers_.size() != config_->layer_num_)
        {
            return InternalError(
                "Create the matmul layer in the attention and ffn attention layers for "
                "the llama model "
                "failed.");
        }

        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            if (!llama_layers_->wq_layers_.at(i) || !llama_layers_->wk_layers_.at(i) ||
                !llama_layers_->wv_layers_.at(i) || !llama_layers_->wo_layers_.at(i))
            {
                return InternalError(
                    "Create the matmul layer in the attention and ffn attention layers for "
                    "the llama model "
                    "failed.");
            }
        }

        if (llama_layers_->w1_layers_.size() != config_->layer_num_ ||
            llama_layers_->w2_layers_.size() != config_->layer_num_ ||
            llama_layers_->w3_layers_.size() != config_->layer_num_)
        {
            return InternalError(
                "Create the matmul layer in the feedforward layers for the llama model "
                "failed.");
        }

        for (int32_t i = 0; i < config_->layer_num_; ++i)
        {
            if (!llama_layers_->w1_layers_.at(i) || !llama_layers_->w2_layers_.at(i) ||
                !llama_layers_->w3_layers_.at(i))
            {
                return InternalError(
                    "Create the matmul layer in the feedforward layers for the llama model "
                    "failed.");
            }
        }

        if (!llama_layers_->rope_layer_)
        {
            return InternalError("Create the rope layer for the llama model failed!");
        }

        if (!llama_layers_->add_layer_)
        {
            return InternalError("Create the add layer for the llama model failed!");
        }

        if (!llama_layers_->mha_layer_)
        {
            return InternalError("Create the mha layer for the llama model failed!");
        }

        if (!llama_layers_->swiglu_layer_)
        {
            return InternalError("Create the SwiGLU layer for the llama model failed!");
        }
        return Success();
    }

    EmbeddingOutput LLama2Model::embedding(const std::vector<int> &tokens) const
    {
        auto input_tokens = get_buffer(ModelBufferType::kInputTokens);
        auto input_embeddings = get_buffer(ModelBufferType::kInputEmbeddings);
        if (input_tokens.size() != tokens.size())
        {
            input_tokens.reshape({static_cast<int32_t>(tokens.size())});
            input_embeddings.reshape({static_cast<int32_t>(tokens.size()), config_->dim_});
        }
        for (int32_t i = 0; i < tokens.size(); ++i)
        {
            input_tokens.index<int32_t>(i) = tokens.at(i);
        }

        auto input_token_num = Tensor(DataType::kDataTypeInt32, static_cast<int32_t>(tokens.size()));
        LOG_IF(FATAL, !llama_layers_->embedding_layer_)
            << "The embedding layer in the llama2 model is null pointer.";
        STATUS_CHECK(llama_layers_->embedding_layer_->forward(input_tokens, input_token_num, input_embeddings));

        EmbeddingOutput output(input_tokens, input_embeddings, input_token_num);
        return output;
    }

    Tensor LLama2Model::fill_input(const Tensor &pos_tensor,
                                   const EmbeddingOutput &embedding_output,
                                   bool is_prompt) const
    {
        const int32_t pos = pos_tensor.index<int32_t>(0);
        auto [input_tokens, input_embeddings, input_token_num] = embedding_output;

        int32_t index = 0;
        if (is_prompt)
        {
            index = pos;
        }
        std::shared_ptr<Buffer> input_emb_buffer =
            std::make_shared<Buffer>(config_->dim_ * sizeof(float), nullptr,
                                     input_embeddings.ptr<float>(index * config_->dim_), true);

        Tensor input(DataType::kDataTypeFp32, config_->dim_);
        input.assign(input_emb_buffer);
        input.set_device_type(device_type_);
        return input;
    }

    void LLama2Model::attention_rms(int32_t layer_idx, const Tensor &input) const
    {
        CHECK(llama_layers_ != nullptr);
        // attn rmsnorm
        Tensor rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
        std::shared_ptr<Layer> rmsnorm_layer = llama_layers_->rmsnorm_layers_.at(layer_idx);
        if (!rmsnorm_layer)
        {
            LOG(FATAL) << "The attention rmsnorm layer is a null pointer in the llama2 model";
        }
        STATUS_CHECK(rmsnorm_layer->forward(input, rmsnorm_output));
    }

    void LLama2Model::attention_qkv(int32_t layer_idx, const Tensor &pos_tensor) const
    {
        CHECK(llama_layers_ != nullptr);
        // kv cache
        Tensor query = this->get_buffer(ModelBufferType::kQuery);
        int32_t pos = pos_tensor.index<int32_t>(0);
        // wq wk wv @ input
        const auto &[key, val] = slice_kv_cache(layer_idx, pos);
        // query
        const auto &query_layer = llama_layers_->wq_layers_.at(layer_idx);
        CHECK_NE(query_layer, nullptr) << "The query layer in the attention block is null pointer.";

        auto rmsnorm_output = get_buffer(ModelBufferType::kOutputRMSNorm);
        STATUS_CHECK(query_layer->forward(rmsnorm_output, query));

        // key
        const auto &key_layer = llama_layers_->wk_layers_.at(layer_idx);
        CHECK_NE(key_layer, nullptr) << "The key layer in the attention block is null pointer.";
        STATUS_CHECK(key_layer->forward(rmsnorm_output, key));
        // value
        const auto &value_layer = llama_layers_->wv_layers_.at(layer_idx);
        CHECK_NE(value_layer, nullptr) << "The value layer in the attention block is null pointer.";
        STATUS_CHECK(value_layer->forward(rmsnorm_output, val));

        // rope
        CHECK_NE(llama_layers_->rope_layer_, nullptr)
            << "The RoPE layer in the attention block is null pointer.";
        STATUS_CHECK(llama_layers_->rope_layer_->forward(
            query, key, pos_tensor, get_buffer(ModelBufferType::kSinCache),
            get_buffer(ModelBufferType::kCosCache), Tensor{}));
    }

    Status LLama2Model::predict(const Tensor &input, const Tensor &pos_tensor,
                                bool is_prompt, int &next) const
    {
        auto status = forward(input, pos_tensor, next);
        if (!status)
        {
            return status;
        }
        next = post_processing(pos_tensor, is_prompt);
        return Success();
    }

    void LLama2Model::attention_mha(int32_t layer_idx, const Tensor &pos_tensor) const
    {
        CHECK(llama_layers_ != nullptr);
        // mha
        Tensor key_cache = get_buffer(ModelBufferType::kKeyCache);
        // VAL = [val1,val2,...val t]
        // output @ VAL = 最终的结果
        Tensor val_cache = get_buffer(ModelBufferType::kValueCache);

        Tensor mha_output = get_buffer(ModelBufferType::kOutputMHA);
        Tensor score_storage = get_buffer(ModelBufferType::kScoreStorage);
        Tensor query = this->get_buffer(ModelBufferType::kQuery);

        const auto &mha_layer = llama_layers_->mha_layer_;
        CHECK_NE(mha_layer, nullptr) << "The multi head attention layer is null pointer.";
        int pos = pos_tensor.index<int32_t>(0);
        std::dynamic_pointer_cast<MultiHeadAttention>(mha_layer)->set_pos(pos);
        std::dynamic_pointer_cast<MultiHeadAttention>(mha_layer)->set_layer_idx(layer_idx);
        STATUS_CHECK(mha_layer->forward(query, score_storage, key_cache, val_cache, mha_output));

        // wo @ attention output
        Tensor attn_output = get_buffer(ModelBufferType::kAttnOutput);
        const auto &wo_layer = llama_layers_->wo_layers_.at(layer_idx);
        CHECK_NE(wo_layer, nullptr) << "The weight output layer is null pointer.";
        STATUS_CHECK(wo_layer->forward(mha_output, attn_output));
    }

    void LLama2Model::feed_forward(int32_t layer_idx, const Tensor &input) const
    {
        CHECK(llama_layers_ != nullptr);
        // residual add
        CHECK_NE(llama_layers_->add_layer_, nullptr)
            << "The add layer in the feedforward block is null pointer";
        STATUS_CHECK(
            llama_layers_->add_layer_->forward(input, get_buffer(ModelBufferType::kAttnOutput), input));

        // ffn rmsnorm
        Tensor ffn_norm_output = get_buffer(ModelBufferType::kFFNRMSNorm);
        const auto &ffn_rmsnorm = llama_layers_->rmsnorm_layers_.at(layer_idx + config_->layer_num_);
        CHECK_NE(ffn_rmsnorm, nullptr)
            << "The final rmsnorm layer in the feedforward block is null pointer";
        STATUS_CHECK(ffn_rmsnorm->forward(input, ffn_norm_output));

        // w1
        Tensor w1_output = get_buffer(ModelBufferType::kW1Output);
        const auto &w1_layer = llama_layers_->w1_layers_.at(layer_idx);
        CHECK_NE(w1_layer, nullptr) << "The w1 layer in the feedforward block is null pointer";
        STATUS_CHECK(w1_layer->forward(ffn_norm_output, w1_output));

        // w3
        Tensor w3_ouput = get_buffer(ModelBufferType::kW3Output);
        const auto &w3_layer = llama_layers_->w3_layers_.at(layer_idx);
        CHECK_NE(w3_layer, nullptr) << "The w3 layer in the feedforward block is null pointer";
        STATUS_CHECK(w3_layer->forward(ffn_norm_output, w3_ouput));

        // SwiGLU
        CHECK_NE(llama_layers_->swiglu_layer_, nullptr)
            << "The swiglu layer in the feedforward block is null pointer";
        STATUS_CHECK(llama_layers_->swiglu_layer_->forward(w1_output, w3_ouput, w1_output));

        // w2
        Tensor w2_output = get_buffer(ModelBufferType::kW2Output);
        const auto &w2_layer = llama_layers_->w2_layers_.at(layer_idx);
        CHECK_NE(w2_layer, nullptr) << "The w2 layer in the feedforward block is null pointer";
        STATUS_CHECK(w2_layer->forward(w1_output, w2_output));

        // residual add
        CHECK_NE(llama_layers_->add_layer_, nullptr)
            << "The add layer in the feedforward block is null pointer";
        STATUS_CHECK(llama_layers_->add_layer_->forward(input, w2_output, input));
    }

    void LLama2Model::cls_logits(const Tensor &input) const
    {
        CHECK(llama_layers_ != nullptr);
        const auto &norm = llama_layers_->rmsnorm_layers_.at(2 * config_->layer_num_);
        CHECK_NE(norm, nullptr);
        STATUS_CHECK(norm->forward(input, input));

        Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        CHECK_NE(llama_layers_->cls_layer_, nullptr);
        STATUS_CHECK(llama_layers_->cls_layer_->forward(input, forward_output));
    }

    int32_t LLama2Model::post_processing(const Tensor &pos, bool is_prompt) const
    {
        Tensor forward_output = get_buffer(ModelBufferType::kForwardOutput);
        const float *forward_logits = forward_output.ptr<float>();

        int32_t next = 0;
        if (is_prompt)
        {
            next = -1;
        }
        else
        {
            next = static_cast<int32_t>(sampler_->sample(forward_logits, forward_output.size(),
                                                         cuda_config_ ? cuda_config_->stream : nullptr));
        }
        return next;
    }

}
