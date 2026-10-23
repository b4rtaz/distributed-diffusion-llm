#ifndef NN_CORE_H
#define NN_CORE_H

#include <chrono>
#include <list>
#include <memory>
#include <cstdint>
#include "nn-quants.hpp"

// primitives

typedef struct {
    NnFloatType floatType;
    NnUint z;
    NnUint y;
    NnUint x;
    NnSize length;
    NnSize nBytes;
    NnSize nBytesXY;
} NnSize3D;

// slices

typedef struct {
    NnUint kvDim0;
    NnSize3D keySize;
    NnSize3D valueSize;
} NnKvCacheSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint d0;
    NnUint n;
    NnSize3D size;
    NnSize3D sliceSize;
} NnRowMatmulSlice;

typedef struct {
    NnFloatType type;
    NnUint nNodes;
    NnUint n;
    NnUint n0;
    NnUint d;
    NnSize3D size;
    NnSize3D sliceSize;
} NnColMatmulSlice;

typedef struct {
    NnUint qDim0;
    NnUint qDimStart;
    NnUint qDimEnd;
    NnUint qShift;
    NnUint kvDim;
    NnUint kvDim0;
    NnUint kvDimStart;
    NnUint sliceDim;
    NnUint seqLen;
    NnUint headDim;
    NnUint nKvHeads;
    float ropeTheta;
    NnSize3D cacheSize;
} NnRopeSlice;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnSize3D attSize;
} NnMultiHeadAttSlice;

typedef struct {
    NnUint logicalSize;
    NnUint logicalStart;
    NnUint logicalEnd;
    NnUint physicalSize;
} NnPaddedShard;

// base enums

enum NnOpCode {
    OP_MERGE_ADD,
    OP_MERGE_SUM,
    OP_EMBEDDING,
    OP_INV_RMS,
    OP_RMS_NORM,
    OP_MATMUL,
    OP_ROPE,
    OP_MULTIHEAD_ATT,
    OP_GELU,
    OP_SILU,
    OP_MUL,
    OP_SCALE,
    OP_CAST,
    OP_MOE_PACK,
    OP_SHIFT,
    OP_SOFTMAX,
    OP_MOE_ROUTE,
    OP_MOE_MERGE,
    OP_DIFFUSION_ATTENTION,
    OP_TIED_EMBEDDING_PARTIAL,
    OP_TANH,
    OP_VOCAB_LOCAL_STATS,
    OP_VOCAB_SAMPLE,
    OP_TIED_EMBEDDING_REDUCE,
    OP_REDUCE_MAX,
    OP_EMBEDDING_MIX_RESCALE,
    OP_EMBEDDING_MIX_FINALIZE,
    OP_COUNT,
};

enum NnOpQuantType {
    // <input>_<weight>_<output>
    F32_F32_F32,
    F32_Q40_F32,
    F32_Q40_Q80,
    F32_F32_Q80,
    Q80_Q80_Q80,
    Q80_Q80_F32,
    Q80_Q40_F32,
    Q80_F32_F32,
};

#define N_OP_CODES OP_COUNT
#define N_OP_QUANTS (Q80_F32_F32 + 1)

enum NnPointerSource {
    SRC_PIPE,
    SRC_BUFFER,
};

enum NnPointerType {
    PNTR_RAW,
    PNTR_BATCH,
    PNTR_BATCHED_SLICE,
    PNTR_BATCHED_GROUP_SLICE,
    PNTR_BATCH_Z,
};

enum NnSyncType {
    SYNC_WITH_ROOT, // whole pipe to all nodes
    SYNC_NODE_SLICES, // my slice of pipe to all nodes
    SYNC_NODE_SLICES_EXCEPT_ROOT, // only workers send slices to root, root does not send
};

enum NnRopeType {
    ROPE_LLAMA = 0,
    ROPE_FALCON = 1,
    ROPE_LLAMA3_1 = 2,
    ROPE_PROPORTIONAL = 3,
};

// base configs

typedef struct {
    char *name;
    NnSize3D size;
} NnPipeConfig;

typedef struct {
    char *name;
    NnSize3D size;
} NnBufferConfig;

typedef struct {
    char *name;
    NnSize3D size;
} NnWeightConfig;

#define NN_NO_SHARED_WEIGHT UINT32_MAX

typedef struct {
    NnPointerSource source;
    NnUint pointerIndex;
    NnPointerType type;
    NnUint groupIndex;
    NnUint groupCount;
} NnPointerConfig;

typedef struct {
    NnOpCode code;
    char *name;
    NnUint index;
    NnPointerConfig input;
    NnPointerConfig output;
    NnSize3D weightSize;
    NnUint weightIndex;
    NnByte *config;
    NnUint configSize;
} NnOpConfig;

typedef struct {
    NnUint pipeIndex;
} NnPreSyncConfig;

typedef struct {
    NnUint pipeIndex;
    NnSyncType syncType;
} NnSyncConfig;

typedef struct  {
    NnUint nOps;
    NnOpConfig *ops;
    NnUint nSyncs;
    NnSyncConfig *syncs;
} NnSegmentConfig;

typedef struct {
    NnUint nBatches;
    NnUint nNodes;
    NnUint nPipes;
    NnPipeConfig *pipes;
    NnUint nPreSyncs;
    NnPreSyncConfig *preSyncs;
} NnNetConfig;

typedef struct {
    NnUint nodeIndex;
    NnUint nBuffers;
    NnBufferConfig *buffers;
    NnUint nWeights;
    NnWeightConfig *weights;
    NnUint nSegments;
    NnSegmentConfig *segments;
} NnNodeConfig;

// op configs

typedef struct {
    NnUint vocabStart;
    NnUint vocabSize; // zero means all rows, for backwards compatibility
    float scale; // zero means 1.0, for backwards compatibility
} NnEmbeddingOpConfig;

typedef struct {
    float epsilon;
    NnUint nColumns;
} NnInvRmsOpConfig;

typedef struct {
    NnUint invRmsBufferIndex;
    NnUint nColumns;
    NnUint scaleFree;
} NnRmsNormOpConfig;

typedef struct {
    NnUint nExperts;
    NnUint nActiveExperts;
    NnUint activeExpertIndexesBufferIndex;
    NnUint routeLayoutBufferIndex;
} NnMatmulOpConfig;

typedef struct {
    NnRopeType type;
    NnUint isQ; // Cannot use `bool` here due to GPU memory alignment
    NnUint positionPipeIndex;
    NnUint ropeCacheBufferIndex;
    float ropeScalingFactor;
    float ropeScalingLowFreqFactor;
    float ropeScalingHighFreqFactor;
    NnUint ropeScalingOrigMaxSeqLen;
    NnUint rotaryDim; // zero means the complete head dimension
    NnRopeSlice slice;
} NnRopeOpConfig;

typedef struct {
    NnUint nHeads;
    NnUint nHeads0;
    NnUint nKvHeads;
    NnUint headDim;
    NnUint seqLen;
    NnUint qSliceD0;
    NnUint kvDim0;
    NnUint positionPipeIndex;
    NnUint queryBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
    NnUint attBufferIndex;
} NnMultiHeadAttOpConfig;

typedef struct {
    // empty
} NnMergeAddOpCodeConfig;

typedef struct {
    NnUint nGroups; // zero retains the original all-Z-to-one reduction
} NnMergeSumOpCodeConfig;

typedef struct {
    // empty
} NnSiluOpCodeConfig;

typedef struct {
    // empty
} NnTanhOpCodeConfig;

typedef struct {
    NnUint multiplierBufferIndex;
    NnUint useWeight; // nonzero reads a shared/op weight vector instead of a buffer
} NnMulOpCodeConfig;

typedef struct {
    NnUint scaleBufferIndex;
    float scalar;
    NnUint useScalar; // nonzero uses scalar and ignores scaleBufferIndex
} NnScaleOpCodeConfig;

typedef struct {
    // empty
} NnCastOpCodeConfig;

typedef struct {
    NnUint routeLayoutBufferIndex;
} NnMoePackOpConfig;

typedef struct {
    NnUint indexPipeIndex;
    NnUint capacity;
} NnShiftOpCodeConfig;

typedef struct {
    // empty
} NnSoftmaxOpCodeConfig;

typedef struct {
    NnUint k;
    NnUint normTopk;
    NnUint indexesBufferIndex;
    NnUint expertScaleBufferIndex;
    NnUint hasExpertScale;
    NnUint expertScaleFromWeight;
    NnUint routeLayoutBufferIndex;
} NnMoeRouteOpConfig;

typedef struct {
    NnUint nActiveExperts;
    NnUint weightsBufferIndex;
    NnUint routeLayoutBufferIndex;
} NnMoeMergeOpConfig;

enum NnDiffusionAttentionMode {
    DIFFUSION_ATTENTION_ENCODER_CAUSAL = 0,
    DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL = 1,
};

typedef struct {
    NnDiffusionAttentionMode mode;
    NnUint nQueryHeads;
    NnUint nKvHeads;
    NnUint globalNQueryHeads;
    NnUint globalNKvHeads;
    NnUint queryHeadStart;
    NnUint kvHeadStart;
    NnUint headDim;
    NnUint slidingWindow; // zero means full attention
    NnUint cacheCapacity;
    float scoreScale;
    NnUint positionPipeIndex;
    NnUint prefixLengthPipeIndex;
    NnUint currentKeyBufferIndex;
    NnUint currentValueBufferIndex;
    NnUint keyCacheBufferIndex;
    NnUint valueCacheBufferIndex;
} NnDiffusionAttentionOpConfig;

typedef struct {
    NnUint temperaturePipeIndex;
    NnUint partialBufferIndex;
    NnUint probabilityBufferIndex;
    NnUint physicalSize;
    NnUint tileSize;
    NnUint tileCount;
    NnUint hiddenSize;
    float logitSoftcap;
} NnTiedEmbeddingPartialOpConfig;

typedef struct {
    NnUint embeddingMixBufferIndex;
    NnUint numeratorBufferIndex;
    NnUint physicalSize;
    NnUint tileCount;
    NnUint hiddenSize;
} NnTiedEmbeddingReduceOpConfig;

typedef struct {
    NnUint divisorPipeIndex;
    NnUint useDivisor;
} NnReduceMaxOpConfig;

typedef struct {
    NnUint vocabularyMaxPipeIndex;
    NnUint globalMaxBufferIndex;
    NnUint nodeIndex;
    NnUint nNodes;
} NnEmbeddingMixRescaleOpConfig;

typedef struct {
    NnUint physicalSize;
    float embeddingScale;
} NnEmbeddingMixFinalizeOpConfig;


#define NN_VOCAB_STATS_WIDTH 5u
#define NN_VOCAB_MICROSHARDS 8u

typedef struct {
    NnUint globalMaxBufferIndex;
    NnUint temperaturePipeIndex;
    NnUint targetTokensPipeIndex;
    NnUint vocabStart;
    NnUint microshardCount;
    NnUint useTargetTokens;
} NnVocabLocalStatsOpConfig;

typedef struct {
    NnUint globalMaxBufferIndex;
    NnUint temperaturePipeIndex;
    NnUint samplingCoinsPipeIndex;
    NnUint localLogitsBufferIndex;
    NnUint vocabStart;
    NnUint microshardStart;
    NnUint microshardCount;
} NnVocabSampleOpConfig;

// utility functions

const char *opCodeToString(NnOpCode code);
const char *opQuantTypeToString(NnOpQuantType type);

NnSize getBytes(NnFloatType floatType, NnSize n);
NnSize getBlockSize(NnFloatType floatType);
NnOpQuantType getOpQuantType(NnFloatType input, NnFloatType weight, NnFloatType output);
NnSize3D size0();
NnSize3D size1D(NnFloatType floatType, NnUint x);
NnSize3D size2D(NnFloatType floatType, NnUint y, NnUint x);
NnSize3D size3D(NnFloatType floatType, NnUint z, NnUint y, NnUint x);
NnPaddedShard paddedShard(NnUint logicalSize, NnUint nNodes, NnUint nodeIndex, NnUint alignment);
NnPointerConfig pointerBatchConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedSliceConfig(NnPointerSource source, NnUint index);
NnPointerConfig pointerBatchedGroupSliceConfig(NnPointerSource source, NnUint index, NnUint groupIndex, NnUint groupCount);
NnPointerConfig pointerBatchZConfig(NnPointerSource source, NnUint index, NnUint zIndex);
NnPointerConfig pointerRawConfig(NnPointerSource source, NnUint index);
bool hasPointerContinuousMemory(NnPointerConfig *config);

void releaseNetConfig(NnNetConfig *netConfig);
void releaseNodeConfig(NnNodeConfig *nodeConfig);

void printNodeRequiredMemory(NnNetConfig *netConfig, NnNodeConfig *nodeConfig);

class Timer {
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
public:
    Timer();
    void reset();
    NnUint elapsedMiliseconds();
    NnUint elapsedMicroseconds();
};

// slicers

NnKvCacheSlice sliceKvCache(NnUint kvDim, NnUint seqLen, NnUint nNodes);
NnRowMatmulSlice sliceRowMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnColMatmulSlice sliceColMatmul(NnFloatType type, NnUint nNodes, NnUint n, NnUint d);
NnRopeSlice sliceRope(NnRopeType type, NnUint qDim, NnUint kvDim, NnUint nKvHeads, NnUint nNodes, NnUint seqLen, NnUint headDim, float ropeTheta, NnUint nodeIndex);
NnMultiHeadAttSlice sliceMultiHeadAtt(NnUint nHeads, NnUint seqLen, NnUint nNodes, NnUint nBatches);

// splitters

NnUint splitRowMatmulWeight(NnRowMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);
NnUint splitColMatmulWeight(NnColMatmulSlice *slice, NnUint nodeIndex, NnByte *weight, NnByte *weight0);

// rope

void fullfillRopeCache(const NnRopeOpConfig *config, float *cache);

#endif
