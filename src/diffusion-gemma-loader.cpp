#include "diffusion-gemma-loader.hpp"
#include "json.hpp"
#include "mmap.hpp"
#include "nn/nn-network.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <set>
#include <stdexcept>

namespace {

const std::uint32_t DDLM_ENDIAN = 0x01020304u;
const std::uint32_t DDLM_VERSION = 1u;
const std::uint32_t DDLM_HEADER_SIZE = 56u;
const std::uint64_t DDLM_ALIGNMENT = 64u;

template <typename T>
T readValue(std::ifstream &file, const char *description) {
    T value;
    if (!file.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error(std::string("Cannot read DDLM ") + description);
    return value;
}

typedef nlohmann::json Json;

const Json &member(const Json &value, const char *name) {
    if (!value.is_object())
        throw std::runtime_error(std::string("DDLM metadata parent for ") + name + " is not an object");
    const Json::const_iterator found = value.find(name);
    if (found == value.end())
        throw std::runtime_error(std::string("DDLM metadata is missing ") + name);
    return found.value();
}

const Json *optionalMember(const Json &value, const char *name) {
    if (!value.is_object())
        throw std::runtime_error("DDLM metadata value is not an object");
    const Json::const_iterator found = value.find(name);
    return found == value.end() ? NULL : &found.value();
}

std::string stringValue(const Json &value, const char *name) {
    const Json &item = member(value, name);
    if (!item.is_string())
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not a string");
    return item.get<std::string>();
}

double numberValue(const Json &value, const char *name) {
    const Json &item = member(value, name);
    if (!item.is_number())
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not a number");
    const double number = item.get<double>();
    if (!std::isfinite(number))
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not finite");
    return number;
}

double optionalNumber(const Json &value, const char *name, double fallback) {
    const Json *item = optionalMember(value, name);
    if (item == NULL) return fallback;
    if (!item->is_number())
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not a number");
    const double number = item->get<double>();
    if (!std::isfinite(number))
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not finite");
    return number;
}

std::uint32_t uintValue(const Json &value, const char *name) {
    const double number = numberValue(value, name);
    if (number < 0 || number > std::numeric_limits<std::uint32_t>::max() || std::floor(number) != number)
        throw std::runtime_error(std::string("DDLM metadata ") + name + " is not a uint32");
    return static_cast<std::uint32_t>(number);
}

float floatValue(const Json &value, const char *name) {
    return static_cast<float>(numberValue(value, name));
}

float optionalFloat(const Json &value, const char *name, float fallback) {
    return static_cast<float>(optionalNumber(value, name, fallback));
}

std::uint64_t checkedMultiply(std::uint64_t left, std::uint64_t right) {
    if (right != 0 && left > std::numeric_limits<std::uint64_t>::max() / right)
        throw std::runtime_error("DDLM tensor element count overflow");
    return left * right;
}

std::uint64_t storedBytes(const std::vector<std::uint64_t> &shape, DiffusionGemmaWeightType type) {
    std::uint64_t count = 1;
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (shape[i] == 0) throw std::runtime_error("DDLM tensor dimension is zero");
        count = checkedMultiply(count, shape[i]);
    }
    if (type == DIFFUSION_GEMMA_F32) return checkedMultiply(count, 4);
    if (type == DIFFUSION_GEMMA_Q40) {
        if (count % 32 != 0) throw std::runtime_error("DDLM Q40 tensor is not block aligned");
        return checkedMultiply(count / 32, 18);
    }
    throw std::runtime_error("Unsupported DDLM tensor type");
}

std::vector<std::uint64_t> shape(std::initializer_list<std::uint64_t> values) {
    return std::vector<std::uint64_t>(values);
}

} // namespace

DiffusionGemmaModelFile::DiffusionGemmaModelFile(const char *path)
    : path_(path == NULL ? "" : path), mappedFile_() {
    if (path_.empty()) throw std::runtime_error("DDLM model path is empty");
    std::ifstream file(path_.c_str(), std::ios::binary);
    if (!file) throw std::runtime_error("Failed to open DDLM model: " + path_);
    char magic[8];
    if (!file.read(magic, sizeof(magic)) || std::memcmp(magic, "DDLM\0\1\0\0", 8) != 0)
        throw std::runtime_error("Invalid DDLM model magic");
    const std::uint32_t endian = readValue<std::uint32_t>(file, "endian marker");
    const std::uint32_t version = readValue<std::uint32_t>(file, "version");
    const std::uint32_t headerSize = readValue<std::uint32_t>(file, "header size");
    const std::uint32_t tensorCount = readValue<std::uint32_t>(file, "tensor count");
    const std::uint64_t metadataSize = readValue<std::uint64_t>(file, "metadata size");
    const std::uint64_t directoryOffset = readValue<std::uint64_t>(file, "directory offset");
    const std::uint64_t dataOffset = readValue<std::uint64_t>(file, "data offset");
    const std::uint64_t declaredSize = readValue<std::uint64_t>(file, "file size");
    file.seekg(0, std::ios::end);
    const std::streamoff actualSize = file.tellg();
    if (endian != DDLM_ENDIAN || version != DDLM_VERSION || headerSize != DDLM_HEADER_SIZE)
        throw std::runtime_error("Unsupported DDLM model format or byte order");
    if (actualSize < 0 || static_cast<std::uint64_t>(actualSize) != declaredSize)
        throw std::runtime_error("DDLM model file-size mismatch");
    if (declaredSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("DDLM model is too large to map");
    if (directoryOffset != headerSize + metadataSize || dataOffset > declaredSize || dataOffset % DDLM_ALIGNMENT != 0)
        throw std::runtime_error("Invalid DDLM model section offsets");
    if (metadataSize > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()))
        throw std::runtime_error("DDLM metadata is too large");
    file.seekg(headerSize, std::ios::beg);
    std::string metadata(static_cast<std::size_t>(metadataSize), '\0');
    if (metadataSize && !file.read(&metadata[0], static_cast<std::streamsize>(metadataSize)))
        throw std::runtime_error("Cannot read DDLM metadata");
    Json root;
    try {
        root = Json::parse(metadata);
    } catch (const Json::exception &error) {
        throw std::runtime_error(std::string("Invalid DDLM metadata JSON: ") + error.what());
    }
    if (stringValue(root, "architecture") != "DiffusionGemmaForBlockDiffusion" || uintValue(root, "format_version") != 1)
        throw std::runtime_error("DDLM does not describe a supported DiffusionGemma model");
    const std::string weightType = stringValue(root, "weight_type");
    if (weightType == "f32") header_.weightType = DIFFUSION_GEMMA_F32;
    else if (weightType == "q40") header_.weightType = DIFFUSION_GEMMA_Q40;
    else throw std::runtime_error("Unsupported DiffusionGemma weight type: " + weightType);
    header_.formatVersion = 1;
    const Json &text = member(root, "text_config");
    header_.vocabSize = uintValue(text, "vocab_size");
    header_.hiddenSize = uintValue(text, "hidden_size");
    header_.layerCount = uintValue(text, "num_hidden_layers");
    header_.maxPositionEmbeddings = uintValue(text, "max_position_embeddings");
    header_.slidingWindow = uintValue(text, "sliding_window");
    header_.canvasLength = uintValue(root, "canvas_length");
    header_.queryHeads = uintValue(text, "num_attention_heads");
    header_.slidingKvHeads = uintValue(text, "num_key_value_heads");
    header_.fullKvHeads = uintValue(text, "num_global_key_value_heads");
    header_.slidingHeadDim = uintValue(text, "head_dim");
    header_.fullHeadDim = uintValue(text, "global_head_dim");
    header_.denseIntermediateSize = uintValue(text, "intermediate_size");
    header_.moeIntermediateSize = uintValue(text, "moe_intermediate_size");
    header_.expertCount = uintValue(text, "num_experts");
    header_.activeExpertCount = uintValue(text, "top_k_experts");
    header_.rmsNormEpsilon = floatValue(text, "rms_norm_eps");
    header_.embeddingScale = floatValue(root, "embedding_scale");
    header_.logitSoftcap = optionalFloat(text, "final_logit_softcapping", 30.0f);
    header_.attentionScoreScale = 1.0f;
    header_.activation = stringValue(text, "hidden_activation");
    if (header_.activation != "gelu_pytorch_tanh") throw std::runtime_error("Unsupported DiffusionGemma activation");
    const Json &ropes = member(text, "rope_parameters");
    const Json &slidingRope = member(ropes, "sliding_attention");
    const Json &fullRope = member(ropes, "full_attention");
    if (stringValue(slidingRope, "rope_type") != "default" || stringValue(fullRope, "rope_type") != "proportional")
        throw std::runtime_error("Unsupported DiffusionGemma RoPE types");
    header_.slidingRope = {DIFFUSION_GEMMA_DEFAULT_ROPE, floatValue(slidingRope, "rope_theta"), 1.0f, 1.0f};
    header_.fullRope = {DIFFUSION_GEMMA_PROPORTIONAL_ROPE, floatValue(fullRope, "rope_theta"), 1.0f, floatValue(fullRope, "partial_rotary_factor")};
    const Json &layerTypes = member(text, "layer_types");
    if (!layerTypes.is_array() || layerTypes.size() != header_.layerCount)
        throw std::runtime_error("DiffusionGemma layer_types count mismatch");
    for (std::size_t i = 0; i < layerTypes.size(); ++i) {
        if (!layerTypes[i].is_string()) throw std::runtime_error("Invalid DiffusionGemma attention type");
        const std::string layerType = layerTypes[i].get<std::string>();
        if (layerType == "sliding_attention") header_.layerTypes.push_back(DIFFUSION_GEMMA_SLIDING_ATTENTION);
        else if (layerType == "full_attention") header_.layerTypes.push_back(DIFFUSION_GEMMA_FULL_ATTENTION);
        else throw std::runtime_error("Unsupported DiffusionGemma attention type");
    }
    const Json &generation = member(root, "generation_config");
    header_.generation.maxDenoisingSteps = static_cast<std::uint32_t>(optionalNumber(generation, "max_denoising_steps", 48));
    header_.generation.maxNewTokens = static_cast<std::uint32_t>(optionalNumber(generation, "max_new_tokens", header_.canvasLength));
    header_.generation.tMin = optionalFloat(generation, "t_min", 0.4f);
    header_.generation.tMax = optionalFloat(generation, "t_max", 0.8f);
    const Json *sampler = optionalMember(generation, "sampler_config");
    header_.generation.entropyBound = sampler == NULL ? 0.1f : optionalFloat(*sampler, "entropy_bound", 0.1f);
    const double stabilityThreshold = optionalNumber(generation, "stability_threshold", 1.0);
    if (stabilityThreshold < 0.0 || stabilityThreshold > UINT32_MAX ||
        std::floor(stabilityThreshold) != stabilityThreshold)
        throw std::runtime_error("Invalid generation stability threshold");
    header_.generation.stabilityThreshold = static_cast<std::uint32_t>(stabilityThreshold);
    header_.generation.confidenceThreshold = optionalFloat(generation, "confidence_threshold", 0.005f);
    header_.generation.padTokenId = static_cast<std::int32_t>(optionalNumber(generation, "pad_token_id", 0));
    const Json *eos = optionalMember(generation, "eos_token_id");
    if (eos != NULL && eos->is_array()) {
        for (std::size_t i = 0; i < eos->size(); ++i) {
            if (!(*eos)[i].is_number())
                throw std::runtime_error("Invalid generation EOS token id");
            const double id = (*eos)[i].get<double>();
            if (id < 0 || id > std::numeric_limits<std::int32_t>::max() || std::floor(id) != id)
                throw std::runtime_error("Invalid generation EOS token id");
            header_.generation.eosTokenIds.push_back(static_cast<std::int32_t>(id));
        }
    } else if (eos != NULL && eos->is_number()) {
        const double id = eos->get<double>();
        if (id < 0 || id > std::numeric_limits<std::int32_t>::max() || std::floor(id) != id)
            throw std::runtime_error("Invalid generation EOS token id");
        header_.generation.eosTokenIds.push_back(static_cast<std::int32_t>(id));
    } else if (eos != NULL) {
        throw std::runtime_error("Invalid generation EOS token id");
    }

    file.seekg(static_cast<std::streamoff>(directoryOffset), std::ios::beg);
    tensors_.reserve(tensorCount);
    for (std::uint32_t i = 0; i < tensorCount; ++i) {
        const std::uint32_t nameSize = readValue<std::uint32_t>(file, "tensor name size");
        const std::uint32_t rawType = readValue<std::uint32_t>(file, "tensor type");
        const std::uint32_t rank = readValue<std::uint32_t>(file, "tensor rank");
        const std::uint32_t flags = readValue<std::uint32_t>(file, "tensor flags");
        const std::uint64_t offset = readValue<std::uint64_t>(file, "tensor offset");
        const std::uint64_t length = readValue<std::uint64_t>(file, "tensor length");
        if (nameSize == 0 || rank == 0 || rank > 8 || flags != 0) throw std::runtime_error("Invalid DDLM tensor directory entry");
        DiffusionGemmaTensor tensor;
        tensor.type = rawType == 0 ? DIFFUSION_GEMMA_F32 : rawType == 2 ? DIFFUSION_GEMMA_Q40 : static_cast<DiffusionGemmaWeightType>(-1);
        tensor.offset = offset;
        tensor.length = length;
        for (std::uint32_t dimension = 0; dimension < rank; ++dimension)
            tensor.shape.push_back(readValue<std::uint64_t>(file, "tensor dimension"));
        tensor.name.resize(nameSize);
        if (!file.read(&tensor.name[0], nameSize)) throw std::runtime_error("Cannot read DDLM tensor name");
        if (tensor.offset < dataOffset || tensor.offset % DDLM_ALIGNMENT != 0 || tensor.length != storedBytes(tensor.shape, tensor.type) || tensor.offset > declaredSize || tensor.length > declaredSize - tensor.offset)
            throw std::runtime_error("Invalid DDLM tensor storage for " + tensor.name);
        if (!tensorIndexes_.insert(std::make_pair(tensor.name, tensors_.size())).second)
            throw std::runtime_error("Duplicate DDLM tensor: " + tensor.name);
        tensors_.push_back(tensor);
    }
    if (static_cast<std::uint64_t>(file.tellg()) > dataOffset) throw std::runtime_error("DDLM tensor directory overlaps tensor data");
    std::vector<DiffusionGemmaTensor> ordered = tensors_;
    std::sort(ordered.begin(), ordered.end(), [](const DiffusionGemmaTensor &a, const DiffusionGemmaTensor &b) { return a.offset < b.offset; });
    std::uint64_t end = dataOffset;
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        if (ordered[i].offset < end) throw std::runtime_error("Overlapping DDLM tensors");
        end = ordered[i].offset + ordered[i].length;
    }
    if (end != declaredSize) throw std::runtime_error("DDLM model has unaccounted trailing bytes");
    validateManifest();
    file.close();
    mappedFile_.reset(new MmapFile());
    openMmapFile(mappedFile_.get(), path_.c_str(), static_cast<std::size_t>(declaredSize));
}

DiffusionGemmaModelFile::~DiffusionGemmaModelFile() {
    if (mappedFile_.get() != NULL)
        closeMmapFile(mappedFile_.get());
}

const DiffusionGemmaTensor &DiffusionGemmaModelFile::tensor(const std::string &name) const {
    const std::unordered_map<std::string, std::size_t>::const_iterator found = tensorIndexes_.find(name);
    if (found == tensorIndexes_.end()) throw std::runtime_error("DDLM model has no tensor named " + name);
    return tensors_[found->second];
}

void DiffusionGemmaModelFile::readTensor(const std::string &name, void *destination, std::size_t bytes) const {
    readTensorBytes(name, 0, destination, bytes);
}

void DiffusionGemmaModelFile::readTensorBytes(const std::string &name, std::uint64_t offset, void *destination, std::size_t bytes) const {
    const DiffusionGemmaTensor &entry = tensor(name);
    if (mappedFile_.get() == NULL || destination == NULL || offset > entry.length || bytes > entry.length - offset)
        throw std::runtime_error("Invalid read range for DDLM tensor " + name);
    const std::uint64_t sourceOffset = entry.offset + offset;
    if (sourceOffset > mappedFile_->size || bytes > mappedFile_->size - sourceOffset)
        throw std::runtime_error("Mapped DDLM tensor range is invalid: " + name);
    if (bytes != 0u)
        std::memcpy(destination,
            static_cast<const NnByte *>(mappedFile_->data) + sourceOffset, bytes);
}

void DiffusionGemmaModelFile::validateManifest() const {
    std::set<std::string> expected;
    const DiffusionGemmaWeightType matrixType = header_.weightType;
    const auto requireTensor = [this, &expected](const std::string &name, const std::vector<std::uint64_t> &dimensions, DiffusionGemmaWeightType type) {
        const DiffusionGemmaTensor &entry = tensor(name);
        if (entry.shape != dimensions || entry.type != type) throw std::runtime_error("DDLM tensor shape/type mismatch: " + name);
        expected.insert(name);
    };
    const std::uint64_t h = header_.hiddenSize, d = header_.denseIntermediateSize, m = header_.moeIntermediateSize;
    const std::uint64_t e = header_.expertCount;
    requireTensor("token_embedding.weight", shape({header_.vocabSize, h}), matrixType);
    requireTensor("token_embedding.mix_weight", shape({h, header_.vocabSize}), matrixType);
    requireTensor("self_conditioning.pre_norm.weight", shape({h}), DIFFUSION_GEMMA_F32);
    requireTensor("self_conditioning.gate_proj.weight", shape({d, h}), matrixType);
    requireTensor("self_conditioning.up_proj.weight", shape({d, h}), matrixType);
    requireTensor("self_conditioning.down_proj.weight", shape({h, d}), matrixType);
    requireTensor("final_norm.weight", shape({h}), DIFFUSION_GEMMA_F32);
    for (std::uint32_t layer = 0; layer < header_.layerCount; ++layer) {
        const std::string prefix = "layers." + std::to_string(layer) + ".";
        const bool sliding = header_.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION;
        const std::uint64_t head = sliding ? header_.slidingHeadDim : header_.fullHeadDim;
        const std::uint64_t kvHeads = sliding ? header_.slidingKvHeads : header_.fullKvHeads;
        const std::uint64_t q = header_.queryHeads * head, kv = kvHeads * head;
        requireTensor(prefix + "experts.down_proj", shape({e, h, m}), matrixType);
        requireTensor(prefix + "experts.gate_up_proj", shape({e, m * 2, h}), matrixType);
        const char *norms[] = {"input_layernorm.weight", "post_attention_layernorm.weight", "post_feedforward_layernorm.weight", "post_feedforward_layernorm_1.weight", "post_feedforward_layernorm_2.weight", "pre_feedforward_layernorm.weight", "pre_feedforward_layernorm_2.weight"};
        for (std::size_t i = 0; i < sizeof(norms) / sizeof(norms[0]); ++i) requireTensor(prefix + norms[i], shape({h}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "decoder_layer_scalar", shape({1}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "encoder_layer_scalar", shape({1}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "mlp.down_proj.weight", shape({h, d}), matrixType);
        requireTensor(prefix + "mlp.gate_proj.weight", shape({d, h}), matrixType);
        requireTensor(prefix + "mlp.up_proj.weight", shape({d, h}), matrixType);
        requireTensor(prefix + "router.per_expert_scale", shape({e}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "router.proj.weight", shape({e, h}), matrixType);
        requireTensor(prefix + "router.scale", shape({h}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "self_attn.k_norm.weight", shape({head}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "self_attn.k_proj.weight", shape({kv, h}), matrixType);
        requireTensor(prefix + "self_attn.o_proj.weight", shape({h, q}), matrixType);
        requireTensor(prefix + "self_attn.q_norm.weight", shape({head}), DIFFUSION_GEMMA_F32);
        requireTensor(prefix + "self_attn.q_proj.weight", shape({q, h}), matrixType);
        if (sliding) requireTensor(prefix + "self_attn.v_proj.weight", shape({kv, h}), matrixType);
    }
    if (expected.size() != tensors_.size()) throw std::runtime_error("DDLM model contains unexpected tensors");
}

namespace {

std::uint64_t tensorElements(const DiffusionGemmaTensor &tensor) {
    std::uint64_t count = 1u;
    for (std::size_t i = 0u; i < tensor.shape.size(); ++i)
        count = checkedMultiply(count, tensor.shape[i]);
    return count;
}

std::uint64_t elementBytes(DiffusionGemmaWeightType type, std::uint64_t elements) {
    if (type == DIFFUSION_GEMMA_F32)
        return checkedMultiply(elements, sizeof(float));
    if (type == DIFFUSION_GEMMA_Q40) {
        if (elements % Q40_BLOCK_SIZE != 0u)
            throw std::runtime_error("Unaligned Q40 element span");
        return checkedMultiply(elements / Q40_BLOCK_SIZE, sizeof(NnBlockQ40));
    }
    throw std::runtime_error("Unsupported DDLM tensor type");
}

void readElements(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaTensor &tensor,
    std::uint64_t elementOffset,
    std::uint64_t elements,
    NnByte *destination) {
    if (elementOffset > tensorElements(tensor) || elements > tensorElements(tensor) - elementOffset)
        throw std::runtime_error("Tensor element slice is out of range: " + tensor.name);
    if (tensor.type == DIFFUSION_GEMMA_Q40 && elementOffset % Q40_BLOCK_SIZE != 0u)
        throw std::runtime_error("Q40 tensor slice starts inside a quantization block: " + tensor.name);
    model.readTensorBytes(tensor.name, elementBytes(tensor.type, elementOffset), destination,
        static_cast<std::size_t>(elementBytes(tensor.type, elements)));
}

void copyRows(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaTensor &tensor,
    NnUint rowStart,
    NnUint logicalRows,
    NnUint physicalRows,
    NnUint rowElements,
    std::vector<NnByte> *output) {
    const std::uint64_t runtimeElements = (std::uint64_t)physicalRows * rowElements;
    output->assign(static_cast<std::size_t>(elementBytes(tensor.type, runtimeElements)), 0u);
    if (logicalRows != 0u)
        readElements(model, tensor, (std::uint64_t)rowStart * rowElements,
            (std::uint64_t)logicalRows * rowElements, output->data());
}

void copyColumns(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaTensor &tensor,
    NnUint columnStart,
    NnUint logicalColumns,
    NnUint physicalColumns,
    NnUint rows,
    std::uint64_t tensorRowOffset,
    std::vector<NnByte> *output) {
    const std::uint64_t canonicalColumns = tensor.shape.back();
    const std::uint64_t runtimeElements = (std::uint64_t)rows * physicalColumns;
    output->assign(static_cast<std::size_t>(elementBytes(tensor.type, runtimeElements)), 0u);
    if (tensor.type == DIFFUSION_GEMMA_F32) {
        const std::size_t rowBytes = static_cast<std::size_t>(logicalColumns * sizeof(float));
        for (NnUint row = 0u; row < rows; ++row) {
            const std::uint64_t sourceElement = tensorRowOffset + (std::uint64_t)row * canonicalColumns + columnStart;
            model.readTensorBytes(tensor.name, sourceElement * sizeof(float),
                output->data() + (std::size_t)row * physicalColumns * sizeof(float), rowBytes);
        }
        return;
    }
    if (columnStart % Q40_BLOCK_SIZE == 0u &&
        logicalColumns == physicalColumns && logicalColumns % Q40_BLOCK_SIZE == 0u) {
        const std::size_t rowBytes = static_cast<std::size_t>(elementBytes(tensor.type, logicalColumns));
        for (NnUint row = 0u; row < rows; ++row) {
            const std::uint64_t sourceElement =
                tensorRowOffset + (std::uint64_t)row * canonicalColumns + columnStart;
            readElements(model, tensor, sourceElement, logicalColumns,
                output->data() + (std::size_t)row * rowBytes);
        }
        return;
    }

    std::vector<NnBlockQ40> source(static_cast<std::size_t>(canonicalColumns / Q40_BLOCK_SIZE));
    std::vector<float> sourceF32(static_cast<std::size_t>(canonicalColumns));
    std::vector<float> padded(physicalColumns);
    const std::size_t destinationRowBytes = static_cast<std::size_t>(elementBytes(tensor.type, physicalColumns));
    for (NnUint row = 0u; row < rows; ++row) {
        readElements(model, tensor, tensorRowOffset + (std::uint64_t)row * canonicalColumns,
            canonicalColumns, reinterpret_cast<NnByte *>(source.data()));
        dequantizeQ40toF32(source.data(), sourceF32.data(), static_cast<NnUint>(canonicalColumns), 1u, 0u);
        std::fill(padded.begin(), padded.end(), 0.0f);
        std::copy(sourceF32.begin() + columnStart, sourceF32.begin() + columnStart + logicalColumns, padded.begin());
        quantizeF32toQ40(padded.data(), reinterpret_cast<NnBlockQ40 *>(output->data() + row * destinationRowBytes),
            physicalColumns, 1u, 0u);
    }
}

void materializeWeight(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaWeightBinding &binding,
    NnSize runtimeBytes,
    std::vector<NnByte> *output) {
    const DiffusionGemmaTensor &tensor = model.tensor(binding.tensorName);
    switch (binding.shard) {
    case DIFFUSION_GEMMA_WEIGHT_REPLICATED:
        if (tensor.length != runtimeBytes)
            throw std::runtime_error("Replicated runtime weight size mismatch: " + binding.runtimeName);
        output->resize(runtimeBytes);
        model.readTensor(tensor.name, output->data(), output->size());
        return;
    case DIFFUSION_GEMMA_WEIGHT_SCALAR_EXPANDED: {
        if (tensor.type != DIFFUSION_GEMMA_F32 || tensorElements(tensor) != 1u || runtimeBytes % sizeof(float) != 0u)
            throw std::runtime_error("Invalid scalar-expanded weight: " + binding.runtimeName);
        float scalar;
        model.readTensor(tensor.name, &scalar, sizeof(scalar));
        output->resize(runtimeBytes);
        std::fill(reinterpret_cast<float *>(output->data()),
            reinterpret_cast<float *>(output->data()) + runtimeBytes / sizeof(float), scalar);
        return;
    }
    case DIFFUSION_GEMMA_WEIGHT_VOCAB_ROWS:
    case DIFFUSION_GEMMA_WEIGHT_ROWS:
    case DIFFUSION_GEMMA_WEIGHT_FULL_KV_ROWS:
    case DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS:
        if (tensor.shape.size() != 2u)
            throw std::runtime_error("Row-sharded tensor is not a matrix: " + tensor.name);
        copyRows(model, tensor, binding.logicalStart, binding.logicalSize, binding.physicalSize,
            static_cast<NnUint>(tensor.shape[1]), output);
        break;
    case DIFFUSION_GEMMA_WEIGHT_COLUMNS:
    case DIFFUSION_GEMMA_WEIGHT_PADDED_COLUMNS:
        if (tensor.shape.size() != 2u)
            throw std::runtime_error("Column-sharded tensor is not a matrix: " + tensor.name);
        copyColumns(model, tensor, binding.logicalStart, binding.logicalSize, binding.physicalSize,
            static_cast<NnUint>(tensor.shape[0]), 0u, output);
        break;
    case DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_ROWS: {
        if (tensor.shape.size() != 3u || binding.splitCount == 0u || binding.splitIndex >= binding.splitCount)
            throw std::runtime_error("Invalid expert row-sharded tensor: " + tensor.name);
        const NnUint experts = static_cast<NnUint>(tensor.shape[0]);
        const NnUint canonicalRows = static_cast<NnUint>(tensor.shape[1]);
        const NnUint rowElements = static_cast<NnUint>(tensor.shape[2]);
        if (canonicalRows % binding.splitCount != 0u)
            throw std::runtime_error("Expert gate/up tensor cannot be split evenly: " + tensor.name);
        const NnUint splitRows = canonicalRows / binding.splitCount;
        const std::uint64_t expertRuntimeElements = (std::uint64_t)binding.physicalSize * rowElements;
        output->assign(static_cast<std::size_t>(elementBytes(tensor.type, expertRuntimeElements * experts)), 0u);
        const std::size_t expertRuntimeBytes = static_cast<std::size_t>(elementBytes(tensor.type, expertRuntimeElements));
        for (NnUint expert = 0u; expert < experts; ++expert) {
            const std::uint64_t sourceRow = (std::uint64_t)expert * canonicalRows +
                binding.splitIndex * splitRows + binding.logicalStart;
            readElements(model, tensor, sourceRow * rowElements,
                (std::uint64_t)binding.logicalSize * rowElements,
                output->data() + expert * expertRuntimeBytes);
        }
        break;
    }
    case DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_COLUMNS: {
        if (tensor.shape.size() != 3u)
            throw std::runtime_error("Invalid expert column-sharded tensor: " + tensor.name);
        const NnUint experts = static_cast<NnUint>(tensor.shape[0]);
        const NnUint rows = static_cast<NnUint>(tensor.shape[1]);
        const NnUint canonicalColumns = static_cast<NnUint>(tensor.shape[2]);
        const std::uint64_t expertCanonicalElements = (std::uint64_t)rows * canonicalColumns;
        const std::uint64_t expertRuntimeElements = (std::uint64_t)rows * binding.physicalSize;
        output->assign(static_cast<std::size_t>(elementBytes(tensor.type, expertRuntimeElements * experts)), 0u);
        const std::size_t expertRuntimeBytes = static_cast<std::size_t>(elementBytes(tensor.type, expertRuntimeElements));
        for (NnUint expert = 0u; expert < experts; ++expert) {
            std::vector<NnByte> oneExpert;
            copyColumns(model, tensor, binding.logicalStart, binding.logicalSize, binding.physicalSize,
                rows, expert * expertCanonicalElements, &oneExpert);
            std::copy(oneExpert.begin(), oneExpert.end(), output->begin() + expert * expertRuntimeBytes);
        }
        break;
    }
    default:
        throw std::runtime_error("Unknown DiffusionGemma weight shard mode");
    }
    if (output->size() != runtimeBytes)
        throw std::runtime_error("Runtime weight size mismatch after sharding: " + binding.runtimeName);
}

} // namespace

void loadDiffusionGemmaWeights(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaGraph &graph,
    NnRootWeightLoader *loader) {
    if (loader == NULL || graph.nodes.empty() || graph.weights.size() != graph.nodes.size())
        throw std::invalid_argument("Invalid DiffusionGemma weight-loading context");
    initQuants();
    std::vector<NnByte> weight;
    for (NnUint nodeIndex = 0u; nodeIndex < graph.nodes.size(); ++nodeIndex) {
        const NnNodeConfig &node = graph.nodes[nodeIndex];
        const std::size_t t = graph.weights[nodeIndex].size();
        for (std::size_t i = 0u; i < t; ++i) {
            const DiffusionGemmaWeightBinding &binding = graph.weights[nodeIndex][i];
            if (binding.weightIndex >= node.nWeights)
                throw std::runtime_error("DiffusionGemma binding references an invalid runtime weight");
            if (nodeIndex == 0u)
                std::printf("💿 Loading %zu/%zu\n", i + 1u, t);
            const NnSize bytes = node.weights[binding.weightIndex].size.nBytes;
            materializeWeight(model, binding, bytes, &weight);
            if (nodeIndex == 0u)
                loader->loadRoot(binding.runtimeName.c_str(), 0u, bytes, weight.data());
            else
                loader->writeWeight(nodeIndex, binding.runtimeName.c_str(), 0u, 0u, bytes, weight.data());
        }
    }
    loader->finish();
}
