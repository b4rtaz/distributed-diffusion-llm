#ifndef DIFFUSION_GEMMA_HPP
#define DIFFUSION_GEMMA_HPP

#include "nn/nn-core.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum DiffusionGemmaWeightType {
    DIFFUSION_GEMMA_F32 = 0,
    DIFFUSION_GEMMA_Q40 = 2,
};

enum DiffusionGemmaAttentionType {
    DIFFUSION_GEMMA_SLIDING_ATTENTION = 0,
    DIFFUSION_GEMMA_FULL_ATTENTION = 1,
};

enum DiffusionGemmaRopeType {
    DIFFUSION_GEMMA_DEFAULT_ROPE = 0,
    DIFFUSION_GEMMA_PROPORTIONAL_ROPE = 1,
};

struct DiffusionGemmaRopeConfig {
    DiffusionGemmaRopeType type;
    float theta;
    float factor;
    float rotaryFactor;
};

struct DiffusionGemmaGenerationConfig {
    std::uint32_t maxDenoisingSteps;
    std::uint32_t maxNewTokens;
    float tMin;
    float tMax;
    float entropyBound;
    std::uint32_t stabilityThreshold;
    float confidenceThreshold;
    std::int32_t padTokenId;
    std::vector<std::int32_t> eosTokenIds;
};

struct DiffusionGemmaHeader {
    std::uint32_t formatVersion;
    DiffusionGemmaWeightType weightType;
    std::uint32_t vocabSize;
    std::uint32_t hiddenSize;
    std::uint32_t layerCount;
    std::uint32_t maxPositionEmbeddings;
    std::uint32_t slidingWindow;
    std::uint32_t canvasLength;
    std::uint32_t queryHeads;
    std::uint32_t slidingKvHeads;
    std::uint32_t fullKvHeads;
    std::uint32_t slidingHeadDim;
    std::uint32_t fullHeadDim;
    std::uint32_t denseIntermediateSize;
    std::uint32_t moeIntermediateSize;
    std::uint32_t expertCount;
    std::uint32_t activeExpertCount;
    float rmsNormEpsilon;
    float embeddingScale;
    float logitSoftcap;
    float attentionScoreScale;
    std::string activation;
    DiffusionGemmaRopeConfig slidingRope;
    DiffusionGemmaRopeConfig fullRope;
    std::vector<DiffusionGemmaAttentionType> layerTypes;
    DiffusionGemmaGenerationConfig generation;
};

struct DiffusionGemmaTensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    DiffusionGemmaWeightType type;
    std::uint64_t offset;
    std::uint64_t length;
};

enum DiffusionGemmaExecutionPlan {
    DIFFUSION_GEMMA_ENCODER_PREFILL = 0,
    DIFFUSION_GEMMA_ENCODER_WITH_LOGITS = 1,
    DIFFUSION_GEMMA_DECODER_DENOISE = 2,
};

struct DiffusionGemmaSegmentRange {
    NnUint first;
    NnUint last;
};

struct DiffusionGemmaGraphPipes {
    NnUint tokens;
    NnUint positions;
    NnUint prefixLength;
    NnUint selfConditioning;
    NnUint temperature;
    NnUint samplingCoins;
    NnUint targetTokens;
    NnUint inputSync;
    NnUint attentionSync;
    NnUint feedForwardSync;
    NnUint localLogits;
    NnUint vocabularyMaxSync;
    NnUint embeddingMixSync;
    NnUint samplerStatsSync;
    NnUint sampledTokensSync;
};

enum DiffusionGemmaWeightShard {
    DIFFUSION_GEMMA_WEIGHT_REPLICATED = 0,
    DIFFUSION_GEMMA_WEIGHT_VOCAB_ROWS = 1,
    DIFFUSION_GEMMA_WEIGHT_ROWS = 2,
    DIFFUSION_GEMMA_WEIGHT_COLUMNS = 3,
    DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS = 4,
    DIFFUSION_GEMMA_WEIGHT_PADDED_COLUMNS = 5,
    DIFFUSION_GEMMA_WEIGHT_FULL_KV_ROWS = 6,
    DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_ROWS = 7,
    DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_COLUMNS = 8,
    DIFFUSION_GEMMA_WEIGHT_SCALAR_EXPANDED = 9,
};

struct DiffusionGemmaWeightBinding {
    std::string runtimeName;
    std::string tensorName;
    NnUint weightIndex;
    DiffusionGemmaWeightShard shard;
    NnUint logicalStart;
    NnUint logicalSize;
    NnUint physicalSize;
    NnUint splitIndex;
    NnUint splitCount;
};

struct DiffusionGemmaGraph {
    NnNetConfig net;
    std::vector<NnNodeConfig> nodes;
    std::vector<std::vector<DiffusionGemmaWeightBinding> > weights;
    DiffusionGemmaGraphPipes pipes;
    DiffusionGemmaSegmentRange encoderPrefill;
    DiffusionGemmaSegmentRange encoderWithLogits;
    DiffusionGemmaSegmentRange decoderDenoise;
    NnUint maxBatchSize;
    NnUint contextCapacity;
};

DiffusionGemmaGraph buildDiffusionGemmaGraph(
    const DiffusionGemmaHeader &header,
    NnUint nNodes,
    NnUint maxBatchSize,
    NnUint contextCapacity);
void releaseDiffusionGemmaGraph(DiffusionGemmaGraph *graph);
DiffusionGemmaSegmentRange diffusionGemmaPlanRange(
    const DiffusionGemmaGraph &graph,
    DiffusionGemmaExecutionPlan plan);

#endif
