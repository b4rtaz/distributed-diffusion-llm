#include "diffusion-gemma.hpp"
#include "diffusion-gemma-generation.hpp"
#include "diffusion-gemma-loader.hpp"
#include "diffusion-gemma-runtime.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"

#include <cassert>
#include <stdexcept>
#include <string>
#include <vector>

static DiffusionGemmaHeader header(bool productionDimensions) {
    DiffusionGemmaHeader h{};
    h.formatVersion = 1u;
    h.weightType = DIFFUSION_GEMMA_F32;
    h.vocabSize = productionDimensions ? 262144u : 64u;
    h.hiddenSize = productionDimensions ? 2816u : 32u;
    h.layerCount = 2u;
    h.maxPositionEmbeddings = productionDimensions ? 262144u : 32u;
    h.slidingWindow = productionDimensions ? 1024u : 4u;
    h.canvasLength = productionDimensions ? 256u : 4u;
    h.queryHeads = productionDimensions ? 16u : 4u;
    h.slidingKvHeads = productionDimensions ? 8u : 2u;
    h.fullKvHeads = productionDimensions ? 2u : 1u;
    h.slidingHeadDim = productionDimensions ? 256u : 8u;
    h.fullHeadDim = productionDimensions ? 512u : 16u;
    h.denseIntermediateSize = productionDimensions ? 2112u : 32u;
    h.moeIntermediateSize = productionDimensions ? 704u : 32u;
    h.expertCount = productionDimensions ? 128u : 2u;
    h.activeExpertCount = productionDimensions ? 8u : 1u;
    h.rmsNormEpsilon = 1e-6f;
    h.embeddingScale = std::sqrt((float)h.hiddenSize);
    h.logitSoftcap = 30.0f;
    h.attentionScoreScale = 1.0f;
    h.activation = "gelu_pytorch_tanh";
    h.slidingRope = {DIFFUSION_GEMMA_DEFAULT_ROPE, 10000.0f, 1.0f, 1.0f};
    h.fullRope = {DIFFUSION_GEMMA_PROPORTIONAL_ROPE, 1000000.0f, 1.0f, 0.25f};
    h.layerTypes.push_back(DIFFUSION_GEMMA_SLIDING_ATTENTION);
    h.layerTypes.push_back(DIFFUSION_GEMMA_FULL_ATTENTION);
    return h;
}

static const DiffusionGemmaWeightBinding &binding(
    const DiffusionGemmaGraph &graph,
    NnUint node,
    const char *name) {
    for (std::size_t i = 0u; i < graph.weights[node].size(); ++i)
        if (graph.weights[node][i].runtimeName == name)
            return graph.weights[node][i];
    throw std::runtime_error(std::string("Missing binding ") + name);
}

static void testProductionShards(NnUint nNodes) {
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(header(true), nNodes, 4u, 16u);
    const NnUint encoderSegments = 10u;
    const NnUint decoderSegments = 11u;
    assert(graph.nodes.size() == nNodes);
    assert(graph.encoderPrefill.first == 0u && graph.encoderPrefill.last == 6u);
    assert(graph.encoderWithLogits.last == 9u);
    assert(graph.decoderDenoise.first == encoderSegments && graph.decoderDenoise.last == 20u);
    for (NnUint node = 0u; node < nNodes; ++node) {
        assert(graph.nodes[node].nSegments == encoderSegments + decoderSegments);
        const DiffusionGemmaWeightBinding &dense = binding(graph, node, "layers.0.mlp.gate_proj.weight");
        const NnUint expectedDense = nNodes <= 2u ? 2112u / nNodes : nNodes == 4u ? 544u : 288u;
        assert(dense.physicalSize == expectedDense);
        const DiffusionGemmaWeightBinding &moe = binding(graph, node, "layers.0.experts.gate");
        const NnUint expectedMoe = nNodes <= 2u ? 704u / nNodes : nNodes == 4u ? 192u : 96u;
        assert(moe.physicalSize == expectedMoe);
        const DiffusionGemmaWeightBinding &fullKv = binding(graph, node, "layers.1.self_attn.k_proj.weight");
        assert(fullKv.shard == DIFFUSION_GEMMA_WEIGHT_FULL_KV_ROWS);
        if (nNodes == 8u)
            assert(fullKv.logicalStart == (node / 4u) * 512u);
    }
    releaseDiffusionGemmaGraph(&graph);
}

static void testCpuGraphInitialization() {
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(header(false), 1u, 256u, 16u);
    NnNetExecution execution(1u, &graph.net);
    execution.setBatchSize(256u);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&graph.net, &graph.nodes[0], &execution), -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, &synchronizer, false);
    const DiffusionGemmaSegmentRange encoder = diffusionGemmaPlanRange(graph, DIFFUSION_GEMMA_ENCODER_PREFILL);
    const DiffusionGemmaSegmentRange decoder = diffusionGemmaPlanRange(graph, DIFFUSION_GEMMA_DECODER_DENOISE);
    assert(encoder.first == 0u && decoder.first > encoder.last);
    devices.clear();
    releaseDiffusionGemmaGraph(&graph);
}

static void testLoadedModel(const char *path) {
    DiffusionGemmaModelFile model(path);
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(
        model.header(), 1u, model.header().canvasLength, model.header().maxPositionEmbeddings);
    NnNetExecution execution(1u, &graph.net);
    execution.setBatchSize(model.header().canvasLength);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&graph.net, &graph.nodes[0], &execution), -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    {
        NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, &synchronizer, false);
        NnRootWeightLoader weights(&executor, NULL, 1u);
        loadDiffusionGemmaWeights(model, graph, &weights);
        RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, NULL);
        std::vector<std::int32_t> tokens(model.header().canvasLength);
        for (NnUint i = 0u; i < model.header().canvasLength; ++i)
            tokens[i] = static_cast<std::int32_t>(i);
        std::vector<std::int32_t> targets(model.header().canvasLength, 3);
        const std::vector<DiffusionGemmaScoredPosition> scores = runtime.score(tokens, targets, 0u, 0u);
        for (NnUint i = 0u; i < model.header().canvasLength; ++i)
            assert(std::fabs(scores[i].probability - 1.0f / model.header().vocabSize) < 1e-7f);
        runtime.resetSelfConditioning();
        const std::vector<DiffusionGemmaSampledPosition> resolved = runtime.denoise(
            tokens, model.header().canvasLength, model.header().canvasLength, 1.0f,
            std::vector<double>(model.header().canvasLength, 0.5), 0u, 1u, false);
        const float *logits = reinterpret_cast<const float *>(execution.pipes[graph.pipes.localLogits]);
        for (NnUint i = 0u; i < model.header().canvasLength * model.header().vocabSize; ++i)
            assert(logits[i] == 0.0f);
        for (NnUint i = 0u; i < model.header().canvasLength; ++i) {
            assert(resolved[i].token == static_cast<std::int32_t>(model.header().vocabSize / 2u));
            assert(resolved[i].argmaxToken == 0);
            assert(std::fabs(resolved[i].entropy - std::log((float)model.header().vocabSize)) < 1e-5f);
        }
    }
    devices.clear();
    releaseDiffusionGemmaGraph(&graph);
}

int main(int argc, char **argv) {
    initQuants();
    testProductionShards(1u);
    testProductionShards(2u);
    testProductionShards(4u);
    testProductionShards(8u);
    testCpuGraphInitialization();
    for (int i = 1; i < argc; ++i)
        testLoadedModel(argv[i]);
    return 0;
}
