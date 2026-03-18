/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "vitAttentionPlugin.h"

#include "common/cudaUtils.h"
#include "common/logger.h"
#include "common/tensor.h"
#include "kernels/contextAttentionKernels/contextFMHARunner.h"
#include "kernels/contextAttentionKernels/utilKernels.h"
#include "plugins/utils/pluginUtils.h"

#ifdef CUTE_DSL_FMHA_ENABLED
#include "kernels/contextAttentionKernels/cuteDslFMHARunner.h"
#endif

#include <cassert>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

using namespace nvinfer1;

namespace trt_edgellm
{
namespace plugins
{

namespace
{
constexpr char const* kATTENTION_PLUGIN_VERSION{"1"};
constexpr char const* kATTENTION_PLUGIN_NAME{"ViTAttentionPlugin"};

// Define the mapping of input and output indices of the ViTAttentionPlugin.
constexpr int32_t kIN_Q_IDX{0};
constexpr int32_t kIN_K_IDX{1};
constexpr int32_t kIN_V_IDX{2};
constexpr int32_t kIN_CU_SEQLENS_IDX{3};
constexpr int32_t kIN_MAX_SEQLEN_CARRIER_IDX{4};
constexpr int32_t kOUT_ATTENTION_IDX{0};

// Reflect the count of Inputs and Outputs of the ViTAttentionPlugin,
// these definitions shall be consistent.
constexpr int32_t kNUM_REQUIRED_INPUTS{5};
constexpr int32_t kNUM_REQUIRED_OUTPUTS{1};

} // namespace

// Static class fields initialization
PluginFieldCollection ViTAttentionPluginCreator::mFieldCollection{};
std::vector<PluginField> ViTAttentionPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ViTAttentionPluginCreator);

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, int32_t numHeads, int32_t headSize)
    : mLayerName(name)
    , mNumHeads(numHeads)
    , mHeadSize(headSize)
{
    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

    bool canImplementFMHA = false;
#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA)
    {
        if (!CuteDslFMHARunner::canImplementViT(mHeadSize, mSMVersion))
        {
            LOG_DEBUG("CuTe DSL ViT FMHA unsupported on SM%d with head_dim=%d, falling back to FMHA_v2", mSMVersion,
                mHeadSize);
            mUseCuteDslFMHA = false;
        }
        else if (CuteDslFMHARunner::loadViTKernelModule())
        {
            canImplementFMHA = true;
            LOG_DEBUG("CuTe DSL ViT FMHA kernel loaded for SM%d", mSMVersion);
        }
        else
        {
            LOG_WARNING("CuTe DSL ViT FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!canImplementFMHA)
#endif
    {
        canImplementFMHA = ContextFMHARunner::canImplement(
            mHeadSize, mSMVersion, mDataType, AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::PADDING);
        if (canImplementFMHA)
        {
            ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
        }
    }

    if (!canImplementFMHA)
    {
        LOG_ERROR("Cannot implement ViTAttentionPlugin configuration. SM: %d, HeadSize: %d, NumHeads: %d", mSMVersion,
            mHeadSize, mNumHeads);
        throw std::runtime_error("Cannot implement the ViTAttentionPlugin configuration.");
    }
}

ViTAttentionPlugin::ViTAttentionPlugin(std::string const& name, std::byte const* data, size_t length)
    : mLayerName(name)
{
    deserializeValue(&data, &length, &mNumHeads);
    deserializeValue(&data, &length, &mHeadSize);

    mSMVersion = getSMVersion();
    applyThorSMRenumberWAR(mSMVersion);

#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA)
    {
        if (!CuteDslFMHARunner::canImplementViT(mHeadSize, mSMVersion))
        {
            LOG_DEBUG("CuTe DSL ViT FMHA unsupported on SM%d with head_dim=%d, falling back to FMHA_v2", mSMVersion,
                mHeadSize);
            mUseCuteDslFMHA = false;
        }
        else if (!CuteDslFMHARunner::loadViTKernelModule())
        {
            LOG_WARNING("CuTe DSL ViT FMHA kernel failed to load, falling back to FMHA_v2");
            mUseCuteDslFMHA = false;
        }
    }
    if (!mUseCuteDslFMHA)
#endif
    {
        ContextFMHARunner::loadContextFMHAKernels(mSMVersion, mDataType);
    }
}

ViTAttentionPlugin::~ViTAttentionPlugin() {}

IPluginV2DynamicExt* ViTAttentionPlugin::clone() const noexcept
{
    ViTAttentionPlugin* plugin = new ViTAttentionPlugin(mLayerName, mNumHeads, mHeadSize);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}

char const* ViTAttentionPlugin::getPluginType() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

char const* ViTAttentionPlugin::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

void ViTAttentionPlugin::setPluginNamespace(char const* pluginNamespace) noexcept
{
    mNamespace = std::string(pluginNamespace);
}

char const* ViTAttentionPlugin::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

int32_t ViTAttentionPlugin::getNbOutputs() const noexcept
{
    // Output attention result.
    return 1;
}

bool ViTAttentionPlugin::supportsFormatCombination(
    int32_t pos, nvinfer1::PluginTensorDesc const* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // Support context/generation phase inputs:
    //      Q tensor (linear FP16) with shape [total_S, H, D]
    //      K tensor (linear FP16) with shape [total_S, H, D]
    //      V tensor (linear FP16) with shape [total_S, H, D]
    //      NOTE: This assumes a head-major layout. The Python export must guarantee this layout.
    //      CuSeqLens tensor (a vector of scalars) with shape [batch_size + 1] and type int32_t.
    //      max_seqlen_carrier tensor with shape [max_seqlen] and type int32_t. Values are ignored.

    // Support context/generation phase outputs:
    //      attention result (linear FP16) with shape [total_S, H, D]
    // Q, K, V, and output all have the same shape [total_S, H, D]
    auto checkQKVO = [this](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kHALF;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 3;
        auto const tensorDim = tensorDesc.dims;
        if (status)
        {
            status &= tensorDim.d[1] == mNumHeads;
            status &= tensorDim.d[2] == mHeadSize;
        }
        return status;
    };

    auto checkCuSeqLens = [](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    auto checkMaxSeqLenCarrier = [](nvinfer1::PluginTensorDesc const& tensorDesc) {
        bool status{true};
        status &= tensorDesc.type == DataType::kINT32;
        status &= tensorDesc.format == TensorFormat::kLINEAR;
        status &= tensorDesc.dims.nbDims == 1;
        return status;
    };

    bool const checkNumIOs = nbInputs == kNUM_REQUIRED_INPUTS && nbOutputs == kNUM_REQUIRED_OUTPUTS;
    if (!checkNumIOs)
    {
        LOG_ERROR(
            "Invalid number of inputs or outputs for the ViTAttentionPlugin '%s'. Expected %d inputs and %d outputs, "
            "but "
            "got %d inputs and %d outputs.",
            mLayerName.c_str(), kNUM_REQUIRED_INPUTS, kNUM_REQUIRED_OUTPUTS, nbInputs, nbOutputs);
        return false;
    }

    bool result{true};

    if (pos < nbInputs)
    {
        switch (pos)
        {
        case kIN_Q_IDX: result = checkQKVO(inOut[0]); break;
        case kIN_K_IDX: result = checkQKVO(inOut[1]); break;
        case kIN_V_IDX: result = checkQKVO(inOut[2]); break;
        case kIN_CU_SEQLENS_IDX: result = checkCuSeqLens(inOut[3]); break;
        case kIN_MAX_SEQLEN_CARRIER_IDX: result = checkMaxSeqLenCarrier(inOut[4]); break;
        default: break;
        }
    }
    else
    {
        int32_t outPos = pos - nbInputs;
        switch (outPos)
        {
        case kOUT_ATTENTION_IDX: result = checkQKVO(inOut[pos]); break;
        default: break;
        }
    }

    return result;
}

// IPluginV2Ext Methods
DataType ViTAttentionPlugin::getOutputDataType([[maybe_unused]] int32_t index,
    [[maybe_unused]] nvinfer1::DataType const* inputTypes, [[maybe_unused]] int32_t nbInputs) const noexcept
{
    // Output[0] (attention) follows Q input dtype (HALF).
    return inputTypes[kIN_Q_IDX];
}

DimsExprs ViTAttentionPlugin::getOutputDimensions(int32_t outputIndex, nvinfer1::DimsExprs const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    // Output[0] has the same shape as Q: [total_S, H, D]
    if (outputIndex == kOUT_ATTENTION_IDX)
    {
        return inputs[kIN_Q_IDX];
    }
    return DimsExprs{};
}

void ViTAttentionPlugin::configurePlugin([[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* in,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::DynamicPluginTensorDesc const* out,
    [[maybe_unused]] int32_t nbOutputs) noexcept
{
    return; // No need to configure anything since we will only use the runtime tensor shapes.
}

size_t ViTAttentionPlugin::getWorkspaceSize([[maybe_unused]] nvinfer1::PluginTensorDesc const* inputs,
    [[maybe_unused]] int32_t nbInputs, [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputs,
    [[maybe_unused]] int32_t nbOutputs) const noexcept
{
    return 0;
}

int32_t ViTAttentionPlugin::enqueue(nvinfer1::PluginTensorDesc const* inputDesc,
    [[maybe_unused]] nvinfer1::PluginTensorDesc const* outputDesc, void const* const* inputs, void* const* outputs,
    [[maybe_unused]] void* workspace, cudaStream_t stream) noexcept
{

    // Construct non-owned tensor objects from I/O data pointers and shapes.
    // Q, K, V inputs in the graph will be in same shape [total_S, H, D].
    PluginTensorDesc const& qInputDesc = inputDesc[kIN_Q_IDX];
    rt::Coords const qkvCoords{qInputDesc.dims};
    rt::Tensor qInputTensor(const_cast<void*>(inputs[kIN_Q_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor kInputTensor(const_cast<void*>(inputs[kIN_K_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);
    rt::Tensor vInputTensor(const_cast<void*>(inputs[kIN_V_IDX]), qkvCoords, rt::DeviceType::kGPU, qInputDesc.type);

    PluginTensorDesc const& cuSeqLensInputDesc = inputDesc[kIN_CU_SEQLENS_IDX];
    rt::Tensor cuSeqLensTensor(const_cast<void*>(inputs[kIN_CU_SEQLENS_IDX]), rt::Coords{cuSeqLensInputDesc.dims},
        rt::DeviceType::kGPU, cuSeqLensInputDesc.type);

    PluginTensorDesc const& maxSeqLenCarrierDesc = inputDesc[kIN_MAX_SEQLEN_CARRIER_IDX];
    int32_t runtimeMaxSeqLen = static_cast<int32_t>(maxSeqLenCarrierDesc.dims.d[0]);

    PluginTensorDesc const& attentionOutputDesc = outputDesc[kOUT_ATTENTION_IDX];
    rt::Tensor attentionOutputTensor(outputs[kOUT_ATTENTION_IDX], rt::Coords{attentionOutputDesc.dims},
        rt::DeviceType::kGPU, attentionOutputDesc.type);

    int32_t runtimeBatchSize = static_cast<int32_t>(cuSeqLensInputDesc.dims.d[0]) - 1;

#ifdef CUTE_DSL_FMHA_ENABLED
    if (mUseCuteDslFMHA)
    {
        int32_t totalSeqLen = static_cast<int32_t>(qInputDesc.dims.d[0]);
        CuteDslFMHARunner runner(mNumHeads, mNumHeads, mHeadSize);
        runner.run(qInputTensor.dataPointer<half>(), kInputTensor.dataPointer<half>(), vInputTensor.dataPointer<half>(),
            attentionOutputTensor.dataPointer<half>(), cuSeqLensTensor.dataPointer<int32_t>(), totalSeqLen,
            runtimeMaxSeqLen, runtimeBatchSize, stream);
    }
    else
#endif
    {
        auto fmhaRunner = ContextFMHARunner(mDataType, runtimeBatchSize, runtimeMaxSeqLen, mNumHeads, mNumHeads,
            mHeadSize, mSMVersion, AttentionInputLayout::SEPARATE_Q_K_V, ContextAttentionMaskType::PADDING, false);

        FusedMultiheadAttentionParamsV2 params{};
        fmhaRunner.setupParams(params);
        params.q_ptr = qInputTensor.dataPointer<half>();
        params.k_ptr = kInputTensor.dataPointer<half>();
        params.v_ptr = vInputTensor.dataPointer<half>();
        params.cu_q_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
        params.cu_kv_seqlens = cuSeqLensTensor.dataPointer<int32_t>();
        params.o_ptr = attentionOutputTensor.dataPointer<half>();

        fmhaRunner.dispatchFMHAKernel(params, stream);
    }

    return 0;
}

size_t ViTAttentionPlugin::getSerializationSize() const noexcept
{
    return sizeof(mNumHeads) + sizeof(mHeadSize);
}

void ViTAttentionPlugin::serialize(void* buffer) const noexcept
{
    std::byte* byteBuffer = static_cast<std::byte*>(buffer);
    serializeValue(&byteBuffer, mNumHeads);
    serializeValue(&byteBuffer, mHeadSize);
}

int32_t ViTAttentionPlugin::initialize() noexcept
{
    return 0;
}

void ViTAttentionPlugin::terminate() noexcept {}

void ViTAttentionPlugin::destroy() noexcept
{
    delete this;
}

ViTAttentionPluginCreator::ViTAttentionPluginCreator()
{
    static std::mutex sMutex;
    std::lock_guard<std::mutex> lock(sMutex);

    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(PluginField("num_heads", nullptr, PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(PluginField("head_size", nullptr, PluginFieldType::kINT32, 1));
    mFieldCollection.nbFields = mPluginAttributes.size();
    mFieldCollection.fields = mPluginAttributes.data();
}

char const* ViTAttentionPluginCreator::getPluginName() const noexcept
{
    return kATTENTION_PLUGIN_NAME;
}

nvinfer1::PluginFieldCollection const* ViTAttentionPluginCreator::getFieldNames() noexcept
{
    return &mFieldCollection;
}

void ViTAttentionPluginCreator::setPluginNamespace(char const* libNamespace) noexcept
{
    mNamespace = libNamespace;
}

char const* ViTAttentionPluginCreator::getPluginNamespace() const noexcept
{
    return mNamespace.c_str();
}

char const* ViTAttentionPluginCreator::getPluginVersion() const noexcept
{
    return kATTENTION_PLUGIN_VERSION;
}

nvinfer1::IPluginV2* ViTAttentionPluginCreator::createPlugin(
    char const* name, nvinfer1::PluginFieldCollection const* fc) noexcept
{
    try
    {
        std::optional<int32_t> numHeads = parsePluginScalarField<int32_t>("num_heads", fc);
        std::optional<int32_t> headSize = parsePluginScalarField<int32_t>("head_size", fc);

        // Enforce Core parameters are specified.
        bool checkRequiredFields = numHeads.has_value() && headSize.has_value();
        if (!checkRequiredFields)
        {
            LOG_ERROR("Missing required ViTAttentionPlugin fields.");
            return nullptr;
        }

        ViTAttentionPlugin* plugin = new ViTAttentionPlugin(std::string(name), numHeads.value(), headSize.value());

        return plugin;
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to create ViTAttentionPlugin: %s", e.what());
    }
    return nullptr;
}

nvinfer1::IPluginV2* ViTAttentionPluginCreator::deserializePlugin(
    char const* name, void const* serialData, size_t serialLength) noexcept
{
    try
    {
        return new ViTAttentionPlugin(name, static_cast<std::byte const*>(serialData), serialLength);
    }
    catch (std::exception const& e)
    {
        LOG_ERROR("Failed to deserialize ViTAttentionPlugin: %s", e.what());
    }
    return nullptr;
}

} // namespace plugins
} // namespace trt_edgellm
