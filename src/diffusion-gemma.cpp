#include "diffusion-gemma.hpp"
#include "nn/nn-config-builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

struct LayerBuffers {
    NnUint query;
    NnUint key;
    NnUint value;
    NnUint attention;
    NnUint quantAttention;
    NnUint keyCache;
    NnUint valueCache;
};

struct NodeBuilder {
    const DiffusionGemmaHeader &h;
    NnUint nNodes;
    NnUint nodeIndex;
    NnUint batches;
    NnUint context;
    NnFloatType weightType;
    NnFloatType transferType;
    NnFloatType matmulInputType;
    const DiffusionGemmaGraphPipes &pipes;
    NnNodeConfigBuilder node;
    std::vector<DiffusionGemmaWeightBinding> bindings;
    std::vector<NnUint> layerWeightBase;
    std::vector<LayerBuffers> layerBuffers;

    NnUint x;
    NnUint norm;
    NnUint quantHidden;
    NnUint invRms;
    NnUint hardEmbedding;
    NnUint localInput;
    NnUint scGate;
    NnUint scUp;
    NnUint scIntermediate;
    NnUint quantDense;
    NnUint localAttention;
    NnUint denseGate;
    NnUint denseUp;
    NnUint denseIntermediate;
    NnUint localDense;
    NnUint routerNorm;
    NnUint routerScores;
    NnUint expertIndexes;
    NnUint expertWeights;
    NnUint expertNorm;
    NnUint expertInput;
    NnUint expertGate;
    NnUint expertUp;
    NnUint expertIntermediate;
    NnUint quantExpertIntermediate;
    NnUint expertOutput;
    NnUint localMoe;
    NnUint branches;
    NnUint branchCombined;
    NnUint ropeSliding;
    NnUint ropeFull;
    NnUint globalVocabularyMax;
    NnUint partialEmbeddingMix;
    NnUint embeddingProbabilities;
    NnUint quantEmbeddingProbabilities;
    NnUint embeddingNumerator;
    NnUint localEmbeddingMix;
    NnUint globalEmbeddingMix;

    NnUint embeddingWeight;
    NnUint embeddingMixWeight;
    NnUint scPreNormWeight;
    NnUint scGateWeight;
    NnUint scUpWeight;
    NnUint scDownWeight;
    NnUint finalNormWeight;

    NodeBuilder(
        const DiffusionGemmaHeader &header,
        NnUint count,
        NnUint index,
        NnUint maxBatch,
        NnUint contextCapacity,
        const DiffusionGemmaGraphPipes &graphPipes)
        : h(header), nNodes(count), nodeIndex(index), batches(maxBatch), context(contextCapacity),
          weightType(header.weightType == DIFFUSION_GEMMA_Q40 ? F_Q40 : F_32),
          transferType(header.weightType == DIFFUSION_GEMMA_Q40 ? F_Q80 : F_32),
          matmulInputType(header.weightType == DIFFUSION_GEMMA_Q40 ? F_Q80 : F_32),
          pipes(graphPipes), node(index) {
        addBuffers();
        addWeights();
    }

    NnUint addWeight(
        const std::string &runtimeName,
        const std::string &tensorName,
        NnSize3D size,
        DiffusionGemmaWeightShard shard,
        NnUint logicalStart = 0u,
        NnUint logicalSize = 0u,
        NnUint physicalSize = 0u,
        NnUint splitIndex = 0u,
        NnUint splitCount = 1u) {
        const NnUint index = node.addWeight(runtimeName.c_str(), size);
        bindings.push_back(DiffusionGemmaWeightBinding{
            runtimeName, tensorName, index, shard, logicalStart, logicalSize, physicalSize, splitIndex, splitCount});
        return index;
    }

    NnUint localVocab() const { return h.vocabSize / nNodes; }
    NnPaddedShard denseShard() const { return paddedShard(h.denseIntermediateSize, nNodes, nodeIndex, 32u); }
    NnPaddedShard moeShard() const { return paddedShard(h.moeIntermediateSize, nNodes, nodeIndex, 32u); }

    NnUint queryDim(NnUint layer) const {
        const NnUint headDim = h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION
            ? h.slidingHeadDim : h.fullHeadDim;
        return (h.queryHeads / nNodes) * headDim;
    }

    NnUint localKvHeads(NnUint layer) const {
        if (h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION)
            return h.slidingKvHeads / nNodes;
        return nNodes <= h.fullKvHeads ? h.fullKvHeads / nNodes : 1u;
    }

    NnUint kvHeadStart(NnUint layer) const {
        if (h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION)
            return nodeIndex * localKvHeads(layer);
        if (nNodes <= h.fullKvHeads)
            return nodeIndex * localKvHeads(layer);
        return nodeIndex / (nNodes / h.fullKvHeads);
    }

    NnUint headDim(NnUint layer) const {
        return h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION
            ? h.slidingHeadDim : h.fullHeadDim;
    }

    NnUint kvDim(NnUint layer) const { return localKvHeads(layer) * headDim(layer); }

    void addBuffers() {
        const NnPaddedShard dense = denseShard();
        const NnPaddedShard moe = moeShard();
        const NnUint maxColumns = std::max(h.queryHeads / nNodes,
            std::max(h.slidingKvHeads / nNodes, 1u));
        x = node.addBuffer("hidden", size2D(F_32, batches, h.hiddenSize));
        norm = node.addBuffer("norm", size2D(F_32, batches, h.hiddenSize));
        quantHidden = node.addBuffer("quant_hidden", size2D(matmulInputType, batches, h.hiddenSize));
        invRms = node.addBuffer("inv_rms", size2D(F_32, batches, maxColumns));
        hardEmbedding = node.addBuffer("hard_embedding", size2D(F_32, batches, h.hiddenSize));
        localInput = node.addBuffer("local_input", size2D(F_32, batches, h.hiddenSize));
        scGate = node.addBuffer("sc_gate", size2D(F_32, batches, dense.physicalSize));
        scUp = node.addBuffer("sc_up", size2D(F_32, batches, dense.physicalSize));
        scIntermediate = node.addBuffer("sc_intermediate", size2D(F_32, batches, dense.physicalSize));
        quantDense = node.addBuffer("quant_dense", size2D(matmulInputType, batches, dense.physicalSize));
        localAttention = node.addBuffer("local_attention", size2D(F_32, batches, h.hiddenSize));
        denseGate = node.addBuffer("dense_gate", size2D(F_32, batches, dense.physicalSize));
        denseUp = node.addBuffer("dense_up", size2D(F_32, batches, dense.physicalSize));
        denseIntermediate = node.addBuffer("dense_intermediate", size2D(F_32, batches, dense.physicalSize));
        localDense = node.addBuffer("local_dense", size2D(F_32, batches, h.hiddenSize));
        routerNorm = node.addBuffer("router_norm", size2D(F_32, batches, h.hiddenSize));
        routerScores = node.addBuffer("router_scores", size2D(F_32, batches, h.expertCount));
        expertIndexes = node.addBuffer("expert_indexes", size2D(F_32, batches, h.activeExpertCount));
        expertWeights = node.addBuffer("expert_weights", size3D(F_32, h.activeExpertCount, batches, 1u));
        expertNorm = node.addBuffer("expert_norm", size2D(F_32, batches, h.hiddenSize));
        expertInput = node.addBuffer("expert_input", size3D(matmulInputType, h.activeExpertCount, batches, h.hiddenSize));
        expertGate = node.addBuffer("expert_gate", size3D(F_32, h.activeExpertCount, batches, moe.physicalSize));
        expertUp = node.addBuffer("expert_up", size3D(F_32, h.activeExpertCount, batches, moe.physicalSize));
        expertIntermediate = node.addBuffer("expert_intermediate", size3D(F_32, h.activeExpertCount, batches, moe.physicalSize));
        quantExpertIntermediate = node.addBuffer("quant_expert_intermediate", size3D(matmulInputType, h.activeExpertCount, batches, moe.physicalSize));
        expertOutput = node.addBuffer("expert_output", size3D(F_32, h.activeExpertCount, batches, h.hiddenSize));
        localMoe = node.addBuffer("local_moe", size2D(F_32, batches, h.hiddenSize));
        branches = node.addBuffer("branches", size3D(F_32, 2u, batches, h.hiddenSize));
        branchCombined = node.addBuffer("branch_combined", size2D(F_32, batches, h.hiddenSize));
        ropeSliding = node.addBuffer("rope_sliding", size2D(F_32, context, h.slidingHeadDim));
        ropeFull = node.addBuffer("rope_full", size2D(F_32, context, h.fullHeadDim));
        const NnUint mixWidth = ((h.hiddenSize + 2u + 31u) / 32u) * 32u;
        globalVocabularyMax = node.addBuffer("global_vocabulary_max", size2D(F_32, batches, 1u));
        partialEmbeddingMix = node.addBuffer("partial_embedding_mix",
            size2D(F_32, batches, mixWidth));
        embeddingProbabilities = node.addBuffer("embedding_probabilities",
            size2D(F_32, batches, localVocab()));
        quantEmbeddingProbabilities = weightType == F_Q40
            ? node.addBuffer("quant_embedding_probabilities",
                size2D(F_Q80, batches, localVocab()))
            : embeddingProbabilities;
        embeddingNumerator = node.addBuffer("embedding_numerator",
            size2D(F_32, batches, h.hiddenSize));
        localEmbeddingMix = node.addBuffer("local_embedding_mix", size2D(F_32, batches, mixWidth));
        globalEmbeddingMix = node.addBuffer("global_embedding_mix", size2D(F_32, batches, mixWidth));

        for (NnUint layer = 0u; layer < h.layerCount; ++layer) {
            const NnUint q = queryDim(layer);
            const NnUint kv = kvDim(layer);
            const NnUint capacity = h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION
                ? std::min(h.slidingWindow, context) : context;
            layerBuffers.push_back(LayerBuffers{
                node.addBuffer(("layer." + std::to_string(layer) + ".query").c_str(), size2D(F_32, batches, q)),
                node.addBuffer(("layer." + std::to_string(layer) + ".key").c_str(), size2D(F_32, batches, kv)),
                node.addBuffer(("layer." + std::to_string(layer) + ".value").c_str(), size2D(F_32, batches, kv)),
                node.addBuffer(("layer." + std::to_string(layer) + ".attention").c_str(), size2D(F_32, batches, q)),
                node.addBuffer(("layer." + std::to_string(layer) + ".quant_attention").c_str(), size2D(matmulInputType, batches, q)),
                node.addBuffer(("layer." + std::to_string(layer) + ".key_cache").c_str(), size1D(F_32, capacity * kv)),
                node.addBuffer(("layer." + std::to_string(layer) + ".value_cache").c_str(), size1D(F_32, capacity * kv)),
            });
        }
    }

    void addWeights() {
        const NnPaddedShard dense = denseShard();
        embeddingWeight = addWeight("token_embedding.weight", "token_embedding.weight",
            size2D(weightType, localVocab(), h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_VOCAB_ROWS,
            nodeIndex * localVocab(), localVocab(), localVocab());
        embeddingMixWeight = addWeight("token_embedding.mix_weight", "token_embedding.mix_weight",
            size2D(weightType, localVocab(), h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_COLUMNS,
            nodeIndex * localVocab(), localVocab(), localVocab());
        scPreNormWeight = addWeight("self_conditioning.pre_norm.weight", "self_conditioning.pre_norm.weight",
            size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        scGateWeight = addWeight("self_conditioning.gate_proj.weight", "self_conditioning.gate_proj.weight",
            size2D(weightType, h.hiddenSize, dense.physicalSize), DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        scUpWeight = addWeight("self_conditioning.up_proj.weight", "self_conditioning.up_proj.weight",
            size2D(weightType, h.hiddenSize, dense.physicalSize), DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        scDownWeight = addWeight("self_conditioning.down_proj.weight", "self_conditioning.down_proj.weight",
            size2D(weightType, dense.physicalSize, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_PADDED_COLUMNS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        finalNormWeight = addWeight("final_norm.weight", "final_norm.weight",
            size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);

        for (NnUint layer = 0u; layer < h.layerCount; ++layer) {
            layerWeightBase.push_back((NnUint)bindings.size());
            addLayerWeights(layer);
        }
    }

    void addLayerWeights(NnUint layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        const NnPaddedShard dense = denseShard();
        const NnPaddedShard moe = moeShard();
        const NnUint qLocal = queryDim(layer);
        const NnUint kvLocal = kvDim(layer);
        const NnUint hd = headDim(layer);
        const bool full = h.layerTypes[layer] == DIFFUSION_GEMMA_FULL_ATTENTION;
        const DiffusionGemmaWeightShard kvShard = full
            ? DIFFUSION_GEMMA_WEIGHT_FULL_KV_ROWS : DIFFUSION_GEMMA_WEIGHT_ROWS;
        const NnUint kvStart = kvHeadStart(layer) * hd;
        addWeight(p + "input_layernorm.weight", p + "input_layernorm.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "self_attn.q_proj.weight", p + "self_attn.q_proj.weight", size2D(weightType, h.hiddenSize, qLocal), DIFFUSION_GEMMA_WEIGHT_ROWS,
            nodeIndex * qLocal, qLocal, qLocal);
        addWeight(p + "self_attn.k_proj.weight", p + "self_attn.k_proj.weight", size2D(weightType, h.hiddenSize, kvLocal), kvShard,
            kvStart, kvLocal, kvLocal);
        if (!full)
            addWeight(p + "self_attn.v_proj.weight", p + "self_attn.v_proj.weight", size2D(weightType, h.hiddenSize, kvLocal), DIFFUSION_GEMMA_WEIGHT_ROWS,
                kvStart, kvLocal, kvLocal);
        addWeight(p + "self_attn.q_norm.weight", p + "self_attn.q_norm.weight", size1D(F_32, hd), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "self_attn.k_norm.weight", p + "self_attn.k_norm.weight", size1D(F_32, hd), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "self_attn.o_proj.weight", p + "self_attn.o_proj.weight", size2D(weightType, qLocal, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_COLUMNS,
            nodeIndex * qLocal, qLocal, qLocal);
        addWeight(p + "post_attention_layernorm.weight", p + "post_attention_layernorm.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "pre_feedforward_layernorm.weight", p + "pre_feedforward_layernorm.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "mlp.gate_proj.weight", p + "mlp.gate_proj.weight", size2D(weightType, h.hiddenSize, dense.physicalSize), DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        addWeight(p + "mlp.up_proj.weight", p + "mlp.up_proj.weight", size2D(weightType, h.hiddenSize, dense.physicalSize), DIFFUSION_GEMMA_WEIGHT_PADDED_ROWS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        addWeight(p + "mlp.down_proj.weight", p + "mlp.down_proj.weight", size2D(weightType, dense.physicalSize, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_PADDED_COLUMNS,
            dense.logicalStart, dense.logicalSize, dense.physicalSize);
        addWeight(p + "post_feedforward_layernorm_1.weight", p + "post_feedforward_layernorm_1.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "pre_feedforward_layernorm_2.weight", p + "pre_feedforward_layernorm_2.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "router.scale", p + "router.scale", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "router.proj.weight", p + "router.proj.weight", size2D(weightType, h.hiddenSize, h.expertCount), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "router.per_expert_scale", p + "router.per_expert_scale", size1D(F_32, h.expertCount), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "experts.gate", p + "experts.gate_up_proj", size3D(weightType, h.expertCount, h.hiddenSize, moe.physicalSize), DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_ROWS,
            moe.logicalStart, moe.logicalSize, moe.physicalSize, 0u, 2u);
        addWeight(p + "experts.up", p + "experts.gate_up_proj", size3D(weightType, h.expertCount, h.hiddenSize, moe.physicalSize), DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_ROWS,
            moe.logicalStart, moe.logicalSize, moe.physicalSize, 1u, 2u);
        addWeight(p + "experts.down_proj", p + "experts.down_proj", size3D(weightType, h.expertCount, moe.physicalSize, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_EXPERT_PADDED_COLUMNS,
            moe.logicalStart, moe.logicalSize, moe.physicalSize);
        addWeight(p + "post_feedforward_layernorm_2.weight", p + "post_feedforward_layernorm_2.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "post_feedforward_layernorm.weight", p + "post_feedforward_layernorm.weight", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_REPLICATED);
        addWeight(p + "encoder_layer_scalar", p + "encoder_layer_scalar", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_SCALAR_EXPANDED);
        addWeight(p + "decoder_layer_scalar", p + "decoder_layer_scalar", size1D(F_32, h.hiddenSize), DIFFUSION_GEMMA_WEIGHT_SCALAR_EXPANDED);
    }

    NnUint lw(NnUint layer, NnUint offset) const { return layerWeightBase[layer] + offset; }

    NnUint layerOffset(NnUint layer, const char *suffix) const {
        const std::string target = "layers." + std::to_string(layer) + "." + suffix;
        for (NnUint i = layerWeightBase[layer]; i < bindings.size(); ++i) {
            if (bindings[i].runtimeName == target)
                return i;
        }
        throw std::runtime_error("Missing graph weight " + target);
    }

    void castForMatmul(NnSegmentConfigBuilder &segment, NnUint input, NnUint output, const std::string &name) {
        segment.addOp(OP_CAST, name.c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, input), pointerBatchConfig(SRC_BUFFER, output), size0(), NnCastOpCodeConfig{});
    }

    void addNorm(
        NnSegmentConfigBuilder &segment,
        NnPointerConfig input,
        NnPointerConfig output,
        NnUint weight,
        const std::string &name,
        NnUint width,
        NnUint columns = 1u,
        bool scaleFree = false) {
        segment.addOp(OP_INV_RMS, (name + ".inv").c_str(), 0u, input,
            pointerBatchConfig(SRC_BUFFER, invRms), size0(), NnInvRmsOpConfig{h.rmsNormEpsilon, columns});
        segment.addOp(OP_RMS_NORM, name.c_str(), 0u, input, output,
            scaleFree ? size0() : size1D(F_32, width / columns),
            scaleFree ? NN_NO_SHARED_WEIGHT : weight,
            NnRmsNormOpConfig{invRms, columns, scaleFree ? 1u : 0u});
    }

    void addMatmul(
        NnSegmentConfigBuilder &segment,
        const std::string &name,
        NnUint input,
        NnUint output,
        NnUint weight,
        NnSize3D weightSize,
        NnUint experts = 0u) {
        segment.addOp(OP_MATMUL, name.c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, input), pointerBatchConfig(SRC_BUFFER, output),
            weightSize, weight, NnMatmulOpConfig{experts, experts == 0u ? 0u : h.activeExpertCount, expertIndexes});
    }

    void addInputSegment(bool decoder) {
        const NnPaddedShard dense = denseShard();
        NnSegmentConfigBuilder s;
        s.addOp(OP_EMBEDDING, "token_embedding.weight", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.tokens), pointerBatchConfig(SRC_BUFFER, hardEmbedding),
            size2D(weightType, localVocab(), h.hiddenSize), embeddingWeight,
            NnEmbeddingOpConfig{nodeIndex * localVocab(), localVocab(), h.embeddingScale});
        if (decoder) {
            addNorm(s, pointerBatchConfig(SRC_PIPE, pipes.selfConditioning), pointerBatchConfig(SRC_BUFFER, norm),
                scPreNormWeight, "self_conditioning.pre_norm.weight", h.hiddenSize);
            castForMatmul(s, norm, quantHidden, "self_conditioning.quantize");
            addMatmul(s, "self_conditioning.gate_proj.weight", quantHidden, scGate, scGateWeight,
                size2D(weightType, h.hiddenSize, dense.physicalSize));
            addMatmul(s, "self_conditioning.up_proj.weight", quantHidden, scUp, scUpWeight,
                size2D(weightType, h.hiddenSize, dense.physicalSize));
            s.addOp(OP_GELU, "self_conditioning.gelu", 0u,
                pointerBatchConfig(SRC_BUFFER, scGate), pointerBatchConfig(SRC_BUFFER, scGate), size0(), NnSiluOpCodeConfig{});
            s.addOp(OP_MUL, "self_conditioning.multiply", 0u,
                pointerBatchConfig(SRC_BUFFER, scGate), pointerBatchConfig(SRC_BUFFER, scIntermediate), size0(),
                NnMulOpCodeConfig{scUp, 0u});
            castForMatmul(s, scIntermediate, quantDense, "self_conditioning.quantize_down");
            addMatmul(s, "self_conditioning.down_proj.weight", quantDense, localInput, scDownWeight,
                size2D(weightType, dense.physicalSize, h.hiddenSize));
            s.addOp(OP_MERGE_ADD, "self_conditioning.add_embedding", 0u,
                pointerBatchConfig(SRC_BUFFER, hardEmbedding), pointerBatchConfig(SRC_BUFFER, localInput), size0(), NnMergeAddOpCodeConfig{});
        } else {
            s.addOp(OP_CAST, "encoder.copy_embedding", 0u,
                pointerBatchConfig(SRC_BUFFER, hardEmbedding), pointerBatchConfig(SRC_BUFFER, localInput), size0(), NnCastOpCodeConfig{});
        }
        s.addOp(OP_CAST, decoder ? "decoder.pack_input" : "encoder.pack_input", 0u,
            pointerBatchConfig(SRC_BUFFER, localInput), pointerBatchedSliceConfig(SRC_PIPE, pipes.inputSync), size0(), NnCastOpCodeConfig{});
        s.addSync(pipes.inputSync, SYNC_NODE_SLICES);
        node.addSegment(s.build());
    }

    void addControlSegment(bool decoder) {
        NnSegmentConfigBuilder s;
        s.addSync(pipes.tokens, SYNC_WITH_ROOT);
        s.addSync(pipes.positions, SYNC_WITH_ROOT);
        s.addSync(pipes.prefixLength, SYNC_WITH_ROOT);
        s.addSync(pipes.temperature, SYNC_WITH_ROOT);
        s.addSync(pipes.targetTokens, SYNC_WITH_ROOT);
        if (decoder) {
            s.addSync(pipes.selfConditioning, SYNC_WITH_ROOT);
            s.addSync(pipes.samplingCoins, SYNC_WITH_ROOT);
        }
        node.addSegment(s.build());
    }

    NnRopeOpConfig ropeConfig(NnUint layer, bool query) const {
        const bool sliding = h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION;
        const DiffusionGemmaRopeConfig &rope = sliding ? h.slidingRope : h.fullRope;
        const NnUint hd = headDim(layer);
        NnRopeSlice slice{};
        slice.qDim0 = queryDim(layer);
        slice.kvDim0 = kvDim(layer);
        slice.seqLen = context;
        slice.headDim = hd;
        slice.ropeTheta = rope.theta;
        slice.cacheSize = size2D(F_32, context, hd);
        return NnRopeOpConfig{
            rope.type == DIFFUSION_GEMMA_PROPORTIONAL_ROPE ? ROPE_PROPORTIONAL : ROPE_FALCON,
            query ? 1u : 0u, pipes.positions, sliding ? ropeSliding : ropeFull,
            rope.factor, 0.0f, 0.0f, 0u,
            (NnUint)(rope.rotaryFactor * hd), slice};
    }

    void finalizeFeedForward(NnSegmentConfigBuilder &s, NnUint layer, bool decoder) {
        s.addOp(OP_MERGE_SUM, "feed_forward.reduce", layer,
            pointerBatchConfig(SRC_PIPE, pipes.feedForwardSync), pointerBatchConfig(SRC_BUFFER, branches),
            size0(), NnMergeSumOpCodeConfig{2u});
        addNorm(s, pointerBatchZConfig(SRC_BUFFER, branches, 0u), pointerBatchZConfig(SRC_BUFFER, branches, 0u),
            layerOffset(layer, "post_feedforward_layernorm_1.weight"),
            "layers." + std::to_string(layer) + ".post_feedforward_layernorm_1.weight", h.hiddenSize);
        addNorm(s, pointerBatchZConfig(SRC_BUFFER, branches, 1u), pointerBatchZConfig(SRC_BUFFER, branches, 1u),
            layerOffset(layer, "post_feedforward_layernorm_2.weight"),
            "layers." + std::to_string(layer) + ".post_feedforward_layernorm_2.weight", h.hiddenSize);
        s.addOp(OP_CAST, "feed_forward.copy_dense", layer,
            pointerBatchZConfig(SRC_BUFFER, branches, 0u), pointerBatchConfig(SRC_BUFFER, branchCombined), size0(), NnCastOpCodeConfig{});
        s.addOp(OP_MERGE_ADD, "feed_forward.add_moe", layer,
            pointerBatchZConfig(SRC_BUFFER, branches, 1u), pointerBatchConfig(SRC_BUFFER, branchCombined), size0(), NnMergeAddOpCodeConfig{});
        addNorm(s, pointerBatchConfig(SRC_BUFFER, branchCombined), pointerBatchConfig(SRC_BUFFER, norm),
            layerOffset(layer, "post_feedforward_layernorm.weight"),
            "layers." + std::to_string(layer) + ".post_feedforward_layernorm.weight", h.hiddenSize);
        s.addOp(OP_MERGE_ADD, "feed_forward.residual", layer,
            pointerBatchConfig(SRC_BUFFER, norm), pointerBatchConfig(SRC_BUFFER, x), size0(), NnMergeAddOpCodeConfig{});
        const NnUint scalar = layerOffset(layer, decoder ? "decoder_layer_scalar" : "encoder_layer_scalar");
        s.addOp(OP_MUL, bindings[scalar].runtimeName.c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, x),
            size1D(F_32, h.hiddenSize), scalar, NnMulOpCodeConfig{0u, 1u});
    }

    void addAttentionSegment(NnUint layer, bool decoder) {
        NnSegmentConfigBuilder s;
        if (layer == 0u) {
            s.addOp(OP_MERGE_SUM, decoder ? "decoder.input.reduce" : "encoder.input.reduce", 0u,
                pointerBatchConfig(SRC_PIPE, pipes.inputSync), pointerBatchConfig(SRC_BUFFER, localInput),
                size0(), NnMergeSumOpCodeConfig{1u});
            if (decoder)
                addNorm(s, pointerBatchConfig(SRC_BUFFER, localInput), pointerBatchConfig(SRC_BUFFER, x),
                    NN_NO_SHARED_WEIGHT, "self_conditioning.post_norm", h.hiddenSize, 1u, true);
            else
                s.addOp(OP_CAST, "encoder.input.copy", 0u,
                    pointerBatchConfig(SRC_BUFFER, localInput), pointerBatchConfig(SRC_BUFFER, x),
                    size0(), NnCastOpCodeConfig{});
        } else {
            finalizeFeedForward(s, layer - 1u, decoder);
        }

        const std::string p = "layers." + std::to_string(layer) + ".";
        const LayerBuffers &b = layerBuffers[layer];
        const NnUint q = queryDim(layer);
        const NnUint kv = kvDim(layer);
        const NnUint hd = headDim(layer);
        const NnUint inputNorm = layerOffset(layer, "input_layernorm.weight");
        addNorm(s, pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, norm), inputNorm,
            p + "input_layernorm.weight", h.hiddenSize);
        castForMatmul(s, norm, quantHidden, p + "quantize_attention_input");
        addMatmul(s, p + "self_attn.q_proj.weight", quantHidden, b.query,
            layerOffset(layer, "self_attn.q_proj.weight"), size2D(weightType, h.hiddenSize, q));
        addMatmul(s, p + "self_attn.k_proj.weight", quantHidden, b.key,
            layerOffset(layer, "self_attn.k_proj.weight"), size2D(weightType, h.hiddenSize, kv));
        if (h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION) {
            addMatmul(s, p + "self_attn.v_proj.weight", quantHidden, b.value,
                layerOffset(layer, "self_attn.v_proj.weight"), size2D(weightType, h.hiddenSize, kv));
        } else {
            s.addOp(OP_CAST, p.c_str(), layer,
                pointerBatchConfig(SRC_BUFFER, b.key), pointerBatchConfig(SRC_BUFFER, b.value), size0(), NnCastOpCodeConfig{});
        }
        addNorm(s, pointerBatchConfig(SRC_BUFFER, b.query), pointerBatchConfig(SRC_BUFFER, b.query),
            layerOffset(layer, "self_attn.q_norm.weight"), p + "self_attn.q_norm.weight", q, h.queryHeads / nNodes);
        addNorm(s, pointerBatchConfig(SRC_BUFFER, b.key), pointerBatchConfig(SRC_BUFFER, b.key),
            layerOffset(layer, "self_attn.k_norm.weight"), p + "self_attn.k_norm.weight", kv, localKvHeads(layer));
        addNorm(s, pointerBatchConfig(SRC_BUFFER, b.value), pointerBatchConfig(SRC_BUFFER, b.value),
            NN_NO_SHARED_WEIGHT, p + "self_attn.v_norm", kv, localKvHeads(layer), true);
        s.addOp(OP_ROPE, (p + "self_attn.q_rope").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, b.query), pointerBatchConfig(SRC_BUFFER, b.query), size0(), ropeConfig(layer, true));
        s.addOp(OP_ROPE, (p + "self_attn.k_rope").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, b.key), pointerBatchConfig(SRC_BUFFER, b.key), size0(), ropeConfig(layer, false));

        const NnUint capacity = h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION
            ? std::min(h.slidingWindow, context) : context;
        s.addOp(OP_DIFFUSION_ATTENTION, (p + (decoder ? "decoder_attention" : "encoder_attention")).c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, b.query), pointerBatchConfig(SRC_BUFFER, b.attention), size0(),
            NnDiffusionAttentionOpConfig{
                decoder ? DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL : DIFFUSION_ATTENTION_ENCODER_CAUSAL,
                h.queryHeads / nNodes, localKvHeads(layer), h.queryHeads,
                h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION ? h.slidingKvHeads : h.fullKvHeads,
                nodeIndex * (h.queryHeads / nNodes), kvHeadStart(layer), hd,
                h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION ? h.slidingWindow : 0u,
                capacity, h.attentionScoreScale, pipes.positions, pipes.prefixLength,
                b.key, b.value, b.keyCache, b.valueCache});
        if (!decoder) {
            s.addOp(OP_SHIFT, (p + "key_cache_update").c_str(), 0u,
                pointerBatchConfig(SRC_BUFFER, b.key), pointerRawConfig(SRC_BUFFER, b.keyCache), size0(),
                NnShiftOpCodeConfig{pipes.positions,
                    h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION ? capacity : 0u});
            s.addOp(OP_SHIFT, (p + "value_cache_update").c_str(), 0u,
                pointerBatchConfig(SRC_BUFFER, b.value), pointerRawConfig(SRC_BUFFER, b.valueCache), size0(),
                NnShiftOpCodeConfig{pipes.positions,
                    h.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION ? capacity : 0u});
        }
        castForMatmul(s, b.attention, b.quantAttention, p + "quantize_attention_output");
        addMatmul(s, p + "self_attn.o_proj.weight", b.quantAttention, localAttention,
            layerOffset(layer, "self_attn.o_proj.weight"), size2D(weightType, q, h.hiddenSize));
        s.addOp(OP_CAST, "attention.pack", layer,
            pointerBatchConfig(SRC_BUFFER, localAttention), pointerBatchedSliceConfig(SRC_PIPE, pipes.attentionSync), size0(), NnCastOpCodeConfig{});
        s.addSync(pipes.attentionSync, SYNC_NODE_SLICES);
        node.addSegment(s.build());
    }

    void addFeedForwardSegment(NnUint layer) {
        const std::string p = "layers." + std::to_string(layer) + ".";
        const NnPaddedShard dense = denseShard();
        const NnPaddedShard moe = moeShard();
        NnSegmentConfigBuilder s;
        s.addOp(OP_MERGE_SUM, "attention.reduce", layer,
            pointerBatchConfig(SRC_PIPE, pipes.attentionSync), pointerBatchConfig(SRC_BUFFER, localInput),
            size0(), NnMergeSumOpCodeConfig{1u});
        addNorm(s, pointerBatchConfig(SRC_BUFFER, localInput), pointerBatchConfig(SRC_BUFFER, norm),
            layerOffset(layer, "post_attention_layernorm.weight"), p + "post_attention_layernorm.weight", h.hiddenSize);
        s.addOp(OP_MERGE_ADD, "attention.residual", layer,
            pointerBatchConfig(SRC_BUFFER, norm), pointerBatchConfig(SRC_BUFFER, x), size0(), NnMergeAddOpCodeConfig{});

        addNorm(s, pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, norm),
            layerOffset(layer, "pre_feedforward_layernorm.weight"), p + "pre_feedforward_layernorm.weight", h.hiddenSize);
        castForMatmul(s, norm, quantHidden, p + "mlp.quantize_input");
        addMatmul(s, p + "mlp.gate_proj.weight", quantHidden, denseGate,
            layerOffset(layer, "mlp.gate_proj.weight"), size2D(weightType, h.hiddenSize, dense.physicalSize));
        addMatmul(s, p + "mlp.up_proj.weight", quantHidden, denseUp,
            layerOffset(layer, "mlp.up_proj.weight"), size2D(weightType, h.hiddenSize, dense.physicalSize));
        s.addOp(OP_GELU, (p + "mlp.gelu").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, denseGate), pointerBatchConfig(SRC_BUFFER, denseGate), size0(), NnSiluOpCodeConfig{});
        s.addOp(OP_MUL, (p + "mlp.multiply").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, denseGate), pointerBatchConfig(SRC_BUFFER, denseIntermediate), size0(),
            NnMulOpCodeConfig{denseUp, 0u});
        castForMatmul(s, denseIntermediate, quantDense, p + "mlp.quantize_down");
        addMatmul(s, p + "mlp.down_proj.weight", quantDense, localDense,
            layerOffset(layer, "mlp.down_proj.weight"), size2D(weightType, dense.physicalSize, h.hiddenSize));

        addNorm(s, pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, routerNorm),
            NN_NO_SHARED_WEIGHT, p + "router.norm", h.hiddenSize, 1u, true);
        const NnUint routerScale = layerOffset(layer, "router.scale");
        s.addOp(OP_MUL, bindings[routerScale].runtimeName.c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, routerNorm), pointerBatchConfig(SRC_BUFFER, routerNorm),
            size1D(F_32, h.hiddenSize), routerScale, NnMulOpCodeConfig{0u, 1u});
        s.addOp(OP_SCALE, (p + "router.root_scale").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, routerNorm), pointerBatchConfig(SRC_BUFFER, routerNorm), size0(),
            NnScaleOpCodeConfig{0u, 1.0f / std::sqrt((float)h.hiddenSize), 1u});
        castForMatmul(s, routerNorm, quantHidden, p + "router.quantize");
        addMatmul(s, p + "router.proj.weight", quantHidden, routerScores,
            layerOffset(layer, "router.proj.weight"), size2D(weightType, h.hiddenSize, h.expertCount));
        s.addOp(OP_SOFTMAX, (p + "router.softmax").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, routerScores), pointerBatchConfig(SRC_BUFFER, routerScores), size0(), NnSoftmaxOpCodeConfig{});
        const NnUint expertScale = layerOffset(layer, "router.per_expert_scale");
        s.addOp(OP_MOE_GATE, bindings[expertScale].runtimeName.c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, routerScores), pointerBatchConfig(SRC_BUFFER, expertWeights),
            size1D(F_32, h.expertCount), expertScale,
            NnMoeGateOpCodeConfig{h.activeExpertCount, 1u, expertIndexes, 0u, 1u, 1u});

        addNorm(s, pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, expertNorm),
            layerOffset(layer, "pre_feedforward_layernorm_2.weight"), p + "pre_feedforward_layernorm_2.weight", h.hiddenSize);
        s.addOp(OP_REPEAT_Z, (p + "experts.repeat").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertNorm), pointerBatchConfig(SRC_BUFFER, expertInput), size0(), NnRepeatZOpCodeConfig{});
        addMatmul(s, p + "experts.gate", expertInput, expertGate,
            layerOffset(layer, "experts.gate"), size3D(weightType, h.expertCount, h.hiddenSize, moe.physicalSize), h.expertCount);
        addMatmul(s, p + "experts.up", expertInput, expertUp,
            layerOffset(layer, "experts.up"), size3D(weightType, h.expertCount, h.hiddenSize, moe.physicalSize), h.expertCount);
        s.addOp(OP_GELU, (p + "experts.gelu").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertGate), pointerBatchConfig(SRC_BUFFER, expertGate), size0(), NnSiluOpCodeConfig{});
        s.addOp(OP_MUL, (p + "experts.multiply").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertGate), pointerBatchConfig(SRC_BUFFER, expertIntermediate), size0(),
            NnMulOpCodeConfig{expertUp, 0u});
        s.addOp(OP_CAST, (p + "experts.quantize_down").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertIntermediate), pointerBatchConfig(SRC_BUFFER, quantExpertIntermediate), size0(), NnCastOpCodeConfig{});
        addMatmul(s, p + "experts.down_proj", quantExpertIntermediate, expertOutput,
            layerOffset(layer, "experts.down_proj"), size3D(weightType, h.expertCount, moe.physicalSize, h.hiddenSize), h.expertCount);
        s.addOp(OP_SCALE, (p + "experts.apply_weights").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertOutput), pointerBatchConfig(SRC_BUFFER, expertOutput), size0(),
            NnScaleOpCodeConfig{expertWeights, 0.0f, 0u});
        s.addOp(OP_MERGE_SUM, (p + "experts.sum").c_str(), 0u,
            pointerBatchConfig(SRC_BUFFER, expertOutput), pointerBatchConfig(SRC_BUFFER, localMoe), size0(), NnMergeSumOpCodeConfig{0u});
        s.addOp(OP_CAST, "feed_forward.pack_dense", layer,
            pointerBatchConfig(SRC_BUFFER, localDense), pointerBatchedGroupSliceConfig(SRC_PIPE, pipes.feedForwardSync, 0u, 2u), size0(), NnCastOpCodeConfig{});
        s.addOp(OP_CAST, "feed_forward.pack_moe", layer,
            pointerBatchConfig(SRC_BUFFER, localMoe), pointerBatchedGroupSliceConfig(SRC_PIPE, pipes.feedForwardSync, 1u, 2u), size0(), NnCastOpCodeConfig{});
        s.addSync(pipes.feedForwardSync, SYNC_NODE_SLICES);
        node.addSegment(s.build());
    }

    void addFinalSegment(bool decoder) {
        NnSegmentConfigBuilder s;
        finalizeFeedForward(s, h.layerCount - 1u, decoder);
        addNorm(s, pointerBatchConfig(SRC_BUFFER, x), pointerBatchConfig(SRC_BUFFER, x),
            finalNormWeight, "final_norm.weight", h.hiddenSize);
        node.addSegment(s.build());
    }

    void addLogitsSegment(bool decoder) {
        NnSegmentConfigBuilder s;
        castForMatmul(s, x, quantHidden, decoder ? "decoder.quantize_logits" : "encoder.quantize_logits");
        s.addOp(OP_MATMUL, "token_embedding.weight", 0u,
            pointerBatchConfig(SRC_BUFFER, quantHidden), pointerBatchConfig(SRC_PIPE, pipes.localLogits),
            size2D(weightType, h.hiddenSize, localVocab()), embeddingWeight,
            NnMatmulOpConfig{0u, 0u, expertIndexes});
        if (!decoder) {
            s.addOp(OP_SCALE, "logits.pre_softcap", 0u,
                pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchConfig(SRC_PIPE, pipes.localLogits), size0(),
                NnScaleOpCodeConfig{0u, 1.0f / h.logitSoftcap, 1u});
            s.addOp(OP_TANH, "logits.tanh", 0u,
                pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchConfig(SRC_PIPE, pipes.localLogits), size0(), NnTanhOpCodeConfig{});
            s.addOp(OP_SCALE, "logits.post_softcap", 0u,
                pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchConfig(SRC_PIPE, pipes.localLogits), size0(),
                NnScaleOpCodeConfig{0u, h.logitSoftcap, 1u});
        }
        node.addSegment(s.build());
    }

    void addVocabularyMaxSegment(bool decoder) {
        NnSegmentConfigBuilder maxima;
        if (decoder) {
            const NnUint mixWidth = ((h.hiddenSize + 2u + 31u) / 32u) * 32u;
            maxima.addOp(OP_TIED_EMBEDDING_PARTIAL, "embedding_mix.probabilities", 0u,
                pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchConfig(SRC_PIPE, pipes.localLogits),
                size0(),
                NnTiedEmbeddingPartialOpConfig{pipes.temperature, partialEmbeddingMix,
                    embeddingProbabilities, mixWidth, localVocab(), 1u,
                    h.hiddenSize, h.logitSoftcap});
            if (weightType == F_Q40) {
                maxima.addOp(OP_CAST, "embedding_mix.quantize_probabilities", 0u,
                    pointerBatchConfig(SRC_BUFFER, embeddingProbabilities),
                    pointerBatchConfig(SRC_BUFFER, quantEmbeddingProbabilities),
                    size0(), NnCastOpCodeConfig{});
            }
            maxima.addOp(OP_MATMUL, "token_embedding.mix_weight", 0u,
                pointerBatchConfig(SRC_BUFFER, quantEmbeddingProbabilities),
                pointerBatchConfig(SRC_BUFFER, embeddingNumerator),
                size2D(weightType, localVocab(), h.hiddenSize),
                embeddingMixWeight, NnMatmulOpConfig{0u, 0u, expertIndexes});
            maxima.addOp(OP_TIED_EMBEDDING_REDUCE, "embedding_mix.reduce_tiles", 0u,
                pointerBatchConfig(SRC_BUFFER, partialEmbeddingMix),
                pointerBatchedSliceConfig(SRC_PIPE, pipes.vocabularyMaxSync), size0(),
                NnTiedEmbeddingReduceOpConfig{localEmbeddingMix, embeddingNumerator,
                    mixWidth, 1u, h.hiddenSize});
            maxima.addSync(pipes.vocabularyMaxSync, SYNC_NODE_SLICES);
            node.addSegment(maxima.build());
            return;
        }
        maxima.addOp(OP_REDUCE_MAX, "embedding_mix.local_max", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchedSliceConfig(SRC_PIPE, pipes.vocabularyMaxSync),
            size0(), NnReduceMaxOpConfig{pipes.temperature, 1u});
        maxima.addSync(pipes.vocabularyMaxSync, SYNC_NODE_SLICES);
        node.addSegment(maxima.build());
    }

    void addVocabularyStatsSegment(bool decoder) {
        const NnUint mixWidth = ((h.hiddenSize + 2u + 31u) / 32u) * 32u;
        NnSegmentConfigBuilder sum;
        sum.addOp(OP_REDUCE_MAX, "embedding_mix.global_max", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.vocabularyMaxSync), pointerBatchConfig(SRC_BUFFER, globalVocabularyMax),
            size0(), NnReduceMaxOpConfig{pipes.temperature, 0u});
        if (decoder) {
            sum.addOp(OP_EMBEDDING_MIX_RESCALE, "embedding_mix.rescale", 0u,
                pointerBatchConfig(SRC_BUFFER, localEmbeddingMix), pointerBatchConfig(SRC_BUFFER, localEmbeddingMix),
                size0(), NnEmbeddingMixRescaleOpConfig{pipes.vocabularyMaxSync,
                    globalVocabularyMax, nodeIndex, nNodes});
            sum.addOp(OP_CAST, "embedding_mix.pack", 0u,
                pointerBatchConfig(SRC_BUFFER, localEmbeddingMix), pointerBatchedSliceConfig(SRC_PIPE, pipes.embeddingMixSync),
                size0(), NnCastOpCodeConfig{});
        }
        sum.addOp(OP_VOCAB_LOCAL_STATS, "sampler.local_stats", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.localLogits), pointerBatchedSliceConfig(SRC_PIPE, pipes.samplerStatsSync),
            size0(), NnVocabLocalStatsOpConfig{globalVocabularyMax, pipes.temperature,
                pipes.targetTokens, nodeIndex * localVocab(), NN_VOCAB_MICROSHARDS / nNodes,
                decoder ? 0u : 1u});
        if (decoder)
            sum.addSync(pipes.embeddingMixSync, SYNC_NODE_SLICES);
        sum.addSync(pipes.samplerStatsSync, SYNC_NODE_SLICES);
        node.addSegment(sum.build());
    }

    void addEmbeddingMixFinalizeSegment() {
        const NnUint mixWidth = ((h.hiddenSize + 2u + 31u) / 32u) * 32u;
        NnSegmentConfigBuilder finalize;
        finalize.addOp(OP_MERGE_SUM, "embedding_mix.reduce", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.embeddingMixSync), pointerBatchConfig(SRC_BUFFER, globalEmbeddingMix),
            size0(), NnMergeSumOpCodeConfig{1u});
        finalize.addOp(OP_EMBEDDING_MIX_FINALIZE, "embedding_mix.finalize", 0u,
            pointerBatchConfig(SRC_BUFFER, globalEmbeddingMix), pointerBatchConfig(SRC_PIPE, pipes.selfConditioning),
            size0(), NnEmbeddingMixFinalizeOpConfig{mixWidth, h.embeddingScale});
        finalize.addOp(OP_VOCAB_SAMPLE, "sampler.sample", 0u,
            pointerBatchConfig(SRC_PIPE, pipes.samplerStatsSync), pointerBatchedSliceConfig(SRC_PIPE, pipes.sampledTokensSync),
            size0(), NnVocabSampleOpConfig{globalVocabularyMax, pipes.temperature,
                pipes.samplingCoins, pipes.localLogits, nodeIndex * localVocab(),
                nodeIndex * (NN_VOCAB_MICROSHARDS / nNodes), NN_VOCAB_MICROSHARDS / nNodes});
        finalize.addSync(pipes.sampledTokensSync, SYNC_NODE_SLICES_EXCEPT_ROOT);
        node.addSegment(finalize.build());
    }

    void addPath(bool decoder) {
        addControlSegment(decoder);
        addInputSegment(decoder);
        for (NnUint layer = 0u; layer < h.layerCount; ++layer) {
            addAttentionSegment(layer, decoder);
            addFeedForwardSegment(layer);
        }
        addFinalSegment(decoder);
        addLogitsSegment(decoder);
        addVocabularyMaxSegment(decoder);
        addVocabularyStatsSegment(decoder);
        if (decoder)
            addEmbeddingMixFinalizeSegment();
    }
};

void validateGraphHeader(const DiffusionGemmaHeader &h, NnUint nNodes, NnUint batches, NnUint context) {
    if (nNodes != 1u && nNodes != 2u && nNodes != 4u && nNodes != 8u)
        throw std::invalid_argument("DiffusionGemma supports 1, 2, 4, or 8 nodes");
    if (batches == 0u || context == 0u || context > h.maxPositionEmbeddings)
        throw std::invalid_argument("Invalid DiffusionGemma batch or context capacity");
    if (h.layerCount == 0u || h.layerTypes.size() != h.layerCount)
        throw std::invalid_argument("DiffusionGemma graph requires at least one described layer");
    if (h.hiddenSize % nNodes != 0u || h.vocabSize % nNodes != 0u ||
        h.queryHeads % nNodes != 0u || h.slidingKvHeads % nNodes != 0u)
        throw std::invalid_argument("DiffusionGemma dimensions do not support this node count");
    if (nNodes > h.fullKvHeads && nNodes % h.fullKvHeads != 0u)
        throw std::invalid_argument("Full-attention KV heads cannot be replicated across this node count");
    if (h.hiddenSize % 32u != 0u || (h.vocabSize / nNodes) % 32u != 0u)
        throw std::invalid_argument("DiffusionGemma quantized dimensions must be block aligned");
    if (h.logitSoftcap <= 0.0f)
        throw std::invalid_argument("DiffusionGemma logit softcap must be positive");
}

} // namespace

DiffusionGemmaGraph buildDiffusionGemmaGraph(
    const DiffusionGemmaHeader &header,
    NnUint nNodes,
    NnUint maxBatchSize,
    NnUint contextCapacity) {
    validateGraphHeader(header, nNodes, maxBatchSize, contextCapacity);
    const NnFloatType transferType = header.weightType == DIFFUSION_GEMMA_Q40 ? F_Q80 : F_32;
    NnNetConfigBuilder net(nNodes, maxBatchSize);
    DiffusionGemmaGraph graph{};
    graph.maxBatchSize = maxBatchSize;
    graph.contextCapacity = contextCapacity;
    graph.pipes.tokens = net.addPipe("tokens", size2D(F_32, maxBatchSize, 1u));
    graph.pipes.positions = net.addPipe("positions", size2D(F_32, maxBatchSize, 1u));
    graph.pipes.prefixLength = net.addPipe("prefix_length", size1D(F_32, 1u));
    graph.pipes.selfConditioning = net.addPipe("self_conditioning", size2D(F_32, maxBatchSize, header.hiddenSize));
    graph.pipes.temperature = net.addPipe("temperature", size1D(F_32, 1u));
    graph.pipes.samplingCoins = net.addPipe("sampling_coins", size2D(F_32, maxBatchSize, 1u));
    graph.pipes.targetTokens = net.addPipe("target_tokens", size2D(F_32, maxBatchSize, 1u));
    graph.pipes.inputSync = net.addPipe("input_sync", size2D(transferType, maxBatchSize, nNodes * header.hiddenSize));
    graph.pipes.attentionSync = net.addPipe("attention_sync", size2D(transferType, maxBatchSize, nNodes * header.hiddenSize));
    graph.pipes.feedForwardSync = net.addPipe("feed_forward_sync", size2D(transferType, maxBatchSize, nNodes * 2u * header.hiddenSize));
    graph.pipes.localLogits = net.addPipe("local_logits", size2D(F_32, maxBatchSize, header.vocabSize / nNodes));
    graph.pipes.vocabularyMaxSync = net.addPipe("vocabulary_max_sync", size2D(F_32, maxBatchSize, nNodes));
    const NnUint mixWidth = ((header.hiddenSize + 2u + 31u) / 32u) * 32u;
    graph.pipes.embeddingMixSync = net.addPipe("embedding_mix_sync", size2D(transferType, maxBatchSize, nNodes * mixWidth));
    graph.pipes.samplerStatsSync = net.addPipe("sampler_stats_sync", size2D(F_32, maxBatchSize,
        NN_VOCAB_MICROSHARDS * NN_VOCAB_STATS_WIDTH));
    graph.pipes.sampledTokensSync = net.addPipe("sampled_tokens_sync", size2D(F_32, maxBatchSize, nNodes));
    net.addPreSync(graph.pipes.tokens);
    net.addPreSync(graph.pipes.positions);
    net.addPreSync(graph.pipes.prefixLength);
    net.addPreSync(graph.pipes.selfConditioning);
    net.addPreSync(graph.pipes.temperature);
    net.addPreSync(graph.pipes.samplingCoins);
    net.addPreSync(graph.pipes.targetTokens);
    graph.net = net.build();

    const NnUint encoderSegments = 2u * header.layerCount + 6u;
    const NnUint decoderSegments = 2u * header.layerCount + 7u;
    graph.encoderPrefill = DiffusionGemmaSegmentRange{0u, encoderSegments - 4u};
    graph.encoderWithLogits = DiffusionGemmaSegmentRange{0u, encoderSegments - 1u};
    graph.decoderDenoise = DiffusionGemmaSegmentRange{encoderSegments, encoderSegments + decoderSegments - 1u};
    graph.nodes.reserve(nNodes);
    graph.weights.reserve(nNodes);
    for (NnUint nodeIndex = 0u; nodeIndex < nNodes; ++nodeIndex) {
        NodeBuilder builder(header, nNodes, nodeIndex, maxBatchSize, contextCapacity, graph.pipes);
        builder.addPath(false);
        builder.addPath(true);
        graph.nodes.push_back(builder.node.build());
        graph.weights.push_back(builder.bindings);
    }
    return graph;
}

void releaseDiffusionGemmaGraph(DiffusionGemmaGraph *graph) {
    if (graph == nullptr)
        return;
    for (std::size_t i = 0u; i < graph->nodes.size(); ++i)
        releaseNodeConfig(&graph->nodes[i]);
    graph->nodes.clear();
    graph->weights.clear();
    releaseNetConfig(&graph->net);
}

DiffusionGemmaSegmentRange diffusionGemmaPlanRange(
    const DiffusionGemmaGraph &graph,
    DiffusionGemmaExecutionPlan plan) {
    if (plan == DIFFUSION_GEMMA_ENCODER_PREFILL)
        return graph.encoderPrefill;
    if (plan == DIFFUSION_GEMMA_ENCODER_WITH_LOGITS)
        return graph.encoderWithLogits;
    if (plan == DIFFUSION_GEMMA_DECODER_DENOISE)
        return graph.decoderDenoise;
    throw std::invalid_argument("Unknown DiffusionGemma execution plan");
}
