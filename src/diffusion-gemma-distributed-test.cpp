#include "diffusion-gemma-loader.hpp"
#include "diffusion-gemma-runtime.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#include <process.h>
#define DDLLM_GETPID _getpid
#else
#include <unistd.h>
#define DDLLM_GETPID getpid
#endif

namespace {

struct NodeSnapshot {
    std::vector<float> tokens;
    std::vector<float> positions;
    std::vector<float> queryNormWeight;
    std::vector<float> keyNormWeight;
    std::vector<float> ropeCache;
    std::vector<float> query;
    std::vector<float> key;
    std::vector<float> value;
    std::vector<float> attention;
    std::vector<float> attentionContribution;
    std::vector<float> attentionProjection;
    std::uint32_t queryWidth;
    std::uint32_t kvWidth;
};

struct DistributedResult {
    std::vector<DiffusionGemmaScoredPosition> scores;
    std::vector<DiffusionGemmaSampledPosition> samples;
    std::vector<DiffusionGemmaSampledPosition> conditionedSamples;
    std::vector<float> hardEmbedding;
    std::vector<NodeSnapshot> nodes;
    std::vector<float> routerScores;
    std::vector<float> expertIndexes;
    std::vector<float> expertWeights;
    std::vector<float> branches;
    std::vector<float> combinedBranches;
    std::vector<float> firstPassHidden;
    std::vector<float> firstPassLogits;
    std::uint32_t localVocab;
    std::uint32_t layerCount;
    bool q80Sync;
};

NnUint bufferIndex(const NnNodeConfig &node, const std::string &name) {
    for (NnUint i = 0u; i < node.nBuffers; ++i)
        if (name == node.buffers[i].name)
            return i;
    throw std::runtime_error("Missing distributed-test buffer: " + name);
}

NnUint pipeIndex(const NnNetConfig &net, const std::string &name) {
    for (NnUint i = 0u; i < net.nPipes; ++i)
        if (name == net.pipes[i].name)
            return i;
    throw std::runtime_error("Missing distributed-test pipe: " + name);
}

std::vector<float> copyBuffer(const NnCpuDevice *cpu, const NnNodeConfig &node, const std::string &name) {
    const NnUint index = bufferIndex(node, name);
    const NnSize count = node.buffers[index].size.nBytes / sizeof(float);
    const float *values = reinterpret_cast<const float *>(cpu->buffers[index]);
    return std::vector<float>(values, values + count);
}

std::vector<float> copyWeight(const NnCpuDevice *cpu, const NnNodeConfig &node, const std::string &name) {
    for (NnUint i = 0u; i < node.nWeights; ++i) {
        if (name == node.weights[i].name) {
            const NnSize count = node.weights[i].size.nBytes / sizeof(float);
            const float *values = reinterpret_cast<const float *>(cpu->weights[i]);
            return std::vector<float>(values, values + count);
        }
    }
    throw std::runtime_error("Missing distributed-test weight: " + name);
}

NodeSnapshot captureNode(
    const NnCpuDevice *cpu,
    const NnNodeConfig &node,
    const NnNetConfig &net,
    const NnNetExecution &execution,
    std::uint32_t queryLength) {
    NodeSnapshot snapshot;
    const float *tokens = reinterpret_cast<const float *>(execution.pipes[pipeIndex(net, "tokens")]);
    const float *positions = reinterpret_cast<const float *>(execution.pipes[pipeIndex(net, "positions")]);
    snapshot.tokens.assign(tokens, tokens + queryLength);
    snapshot.positions.assign(positions, positions + queryLength);
    snapshot.queryNormWeight = copyWeight(cpu, node, "layers.0.self_attn.q_norm.weight");
    snapshot.keyNormWeight = copyWeight(cpu, node, "layers.0.self_attn.k_norm.weight");
    snapshot.ropeCache = copyBuffer(cpu, node,
        snapshot.queryNormWeight.size() == 512u ? "rope_full" : "rope_sliding");
    snapshot.query = copyBuffer(cpu, node, "layer.0.query");
    snapshot.key = copyBuffer(cpu, node, "layer.0.key");
    snapshot.value = copyBuffer(cpu, node, "layer.0.value");
    snapshot.attention = copyBuffer(cpu, node, "layer.0.attention");
    snapshot.attentionContribution = copyBuffer(cpu, node, "local_attention");
    snapshot.attentionProjection = copyBuffer(cpu, node, "local_input");
    snapshot.queryWidth = static_cast<std::uint32_t>(snapshot.query.size() / queryLength);
    snapshot.kvWidth = static_cast<std::uint32_t>(snapshot.key.size() / queryLength);
    return snapshot;
}

std::uint32_t contextCapacity(const NnNodeConfig &node) {
    for (NnUint i = 0u; i < node.nBuffers; ++i)
        if (std::string(node.buffers[i].name) == "rope_full")
            return node.buffers[i].size.y;
    throw std::runtime_error("Worker graph has no context-capacity buffer");
}

void runWorker(std::uint32_t port, std::exception_ptr *error, NodeSnapshot *snapshot) {
    try {
        std::unique_ptr<NnNetwork> network = NnNetwork::serve("127.0.0.1", static_cast<int>(port));
        NnWorkerConfigReader reader(network.get());
        NnNetConfig net = reader.readNet();
        NnNodeConfig node = reader.readNode();
        std::unique_ptr<NnNetConfig, void (*)(NnNetConfig *)> netGuard(&net, releaseNetConfig);
        std::unique_ptr<NnNodeConfig, void (*)(NnNodeConfig *)> nodeGuard(&node, releaseNodeConfig);
        NnNetExecution execution(1u, &net);
        std::vector<NnExecutorDevice> devices;
        NnCpuDevice *cpu = new NnCpuDevice(&net, &node, &execution);
        devices.push_back(NnExecutorDevice(cpu, -1, -1));
        NnNetworkNodeSynchronizer synchronizer(network.get(), &execution, &net, &node);
        NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
        NnWorkerWeightReader weights(&executor, network.get());
        weights.read();
        WorkerDiffusionGemmaRuntime runtime(
            &execution, &executor, network.get(), node.nSegments, contextCapacity(node));
        std::uint32_t forwards = 0u;
        while (!runtime.isFinished) {
            if (runtime.tryRun(0ul) && !runtime.isFinished) {
                ++forwards;
                if (forwards == 2u)
                    *snapshot = captureNode(cpu, node, net, execution, execution.batchSize);
            }
        }
    } catch (...) {
        *error = std::current_exception();
    }
}

DistributedResult runModel(
    const char *path,
    NnUint nNodes,
    std::uint32_t portBase,
    std::uint32_t queryLength = 0u,
    std::uint32_t contextCapacityOverride = 0u) {
    DiffusionGemmaModelFile model(path);
    if (queryLength == 0u)
        queryLength = model.header().canvasLength;
    const std::uint32_t context = contextCapacityOverride == 0u
        ? model.header().maxPositionEmbeddings
        : contextCapacityOverride;
    if (queryLength > model.header().canvasLength || queryLength > context / 2u)
        throw std::invalid_argument("Invalid distributed-test query or context size");
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(
        model.header(), nNodes, queryLength, context);
    std::unique_ptr<DiffusionGemmaGraph, void (*)(DiffusionGemmaGraph *)> graphGuard(
        &graph, releaseDiffusionGemmaGraph);

    std::vector<std::thread> workers;
    std::vector<std::exception_ptr> workerErrors(nNodes > 1u ? nNodes - 1u : 0u);
    std::vector<NodeSnapshot> workerSnapshots(workerErrors.size());
    std::vector<std::string> hostStorage(workerErrors.size(), "127.0.0.1");
    std::vector<char *> hosts(workerErrors.size());
    std::vector<NnUint> ports(workerErrors.size());
    for (std::size_t i = 0u; i < workerErrors.size(); ++i) {
        hosts[i] = const_cast<char *>(hostStorage[i].c_str());
        ports[i] = portBase + static_cast<std::uint32_t>(i);
        workers.push_back(std::thread(runWorker, ports[i], &workerErrors[i], &workerSnapshots[i]));
    }
    if (!workers.empty())
        std::this_thread::sleep_for(std::chrono::milliseconds(250));

    DistributedResult result;
    std::exception_ptr rootError;
    try {
        NnNetExecution execution(1u, &graph.net);
        std::unique_ptr<NnNetwork> network;
        std::unique_ptr<NnNodeSynchronizer> synchronizer;
        if (nNodes == 1u) {
            synchronizer.reset(new NnFakeNodeSynchronizer());
        } else {
            network = NnNetwork::connect(nNodes - 1u, hosts.data(), ports.data());
            NnRootConfigWriter writer(network.get());
            writer.writeToWorkers(&graph.net, graph.nodes.data());
            synchronizer.reset(
                new NnNetworkNodeSynchronizer(network.get(), &execution, &graph.net, &graph.nodes[0]));
        }
        std::vector<NnExecutorDevice> devices;
        NnCpuDevice *cpu = new NnCpuDevice(&graph.net, &graph.nodes[0], &execution);
        devices.push_back(NnExecutorDevice(cpu, -1, -1));
        NnExecutor executor(
            &graph.net, &graph.nodes[0], &devices, &execution, synchronizer.get(), false);
        NnRootWeightLoader weights(&executor, network.get(), nNodes);
        loadDiffusionGemmaWeights(model, graph, &weights);
        RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, network.get());

        std::vector<std::int32_t> tokens(queryLength);
        std::vector<std::int32_t> targets(queryLength);
        std::vector<double> coins(queryLength);
        for (std::uint32_t i = 0u; i < queryLength; ++i) {
            tokens[i] = static_cast<std::int32_t>((2u + i * 15u) % model.header().vocabSize);
            targets[i] = static_cast<std::int32_t>((7u + i * 8u) % model.header().vocabSize);
            coins[i] = static_cast<double>(i + 1u) / static_cast<double>(queryLength + 1u);
        }
        result.scores = runtime.score(tokens, targets, 0u, 0u);
        runtime.resetSelfConditioning();
        result.samples = runtime.denoise(
            targets, queryLength, queryLength, 0.8f, coins, 0u, 2u, false);
        result.hardEmbedding = copyBuffer(cpu, graph.nodes[0], "hard_embedding");
        result.nodes.push_back(captureNode(cpu, graph.nodes[0], graph.net, execution, queryLength));
        result.routerScores = copyBuffer(cpu, graph.nodes[0], "router_scores");
        result.expertIndexes = copyBuffer(cpu, graph.nodes[0], "expert_indexes");
        result.expertWeights = copyBuffer(cpu, graph.nodes[0], "expert_weights");
        result.branches = copyBuffer(cpu, graph.nodes[0], "branches");
        result.combinedBranches = copyBuffer(cpu, graph.nodes[0], "branch_combined");
        result.localVocab = model.header().vocabSize / nNodes;
        result.layerCount = model.header().layerCount;
        result.firstPassHidden = copyBuffer(cpu, graph.nodes[0], "hidden");
        result.firstPassLogits = copyBuffer(cpu, graph.nodes[0], "local_logits");
        result.firstPassLogits.resize(static_cast<std::size_t>(queryLength) * result.localVocab);
        result.conditionedSamples = runtime.denoise(
            targets, queryLength, queryLength, 0.8f, coins, 0u, 1u, true);
        result.q80Sync = model.header().weightType == DIFFUSION_GEMMA_Q40;
        runtime.finish();
        for (std::size_t i = 0u; i < workers.size(); ++i)
            workers[i].join();
        result.nodes.insert(result.nodes.end(), workerSnapshots.begin(), workerSnapshots.end());
    } catch (...) {
        rootError = std::current_exception();
    }

    for (std::size_t i = 0u; i < workers.size(); ++i)
        if (workers[i].joinable())
            workers[i].join();
    if (rootError)
        std::rethrow_exception(rootError);
    for (std::size_t i = 0u; i < workerErrors.size(); ++i)
        if (workerErrors[i])
            std::rethrow_exception(workerErrors[i]);
    return result;
}

void requireEqual(const char *name, std::size_t index, std::int32_t expected, std::int32_t actual) {
    if (expected == actual)
        return;
    std::fprintf(stderr, "%s[%zu]: expected %d, got %d\n", name, index, expected, actual);
    throw std::runtime_error(std::string("Distributed mismatch: ") + name);
}

void requireNear(const char *name, std::size_t index, float expected, float actual, float tolerance) {
    const float difference = std::fabs(expected - actual);
    if (difference < tolerance)
        return;
    std::fprintf(stderr, "%s[%zu]: expected %.9g, got %.9g, difference %.9g (limit %.9g)\n",
        name, index, expected, actual, difference, tolerance);
    throw std::runtime_error(std::string("Distributed mismatch: ") + name);
}

void requireSize(const char *name, std::size_t expected, std::size_t actual) {
    if (expected == actual)
        return;
    std::fprintf(stderr, "%s: expected %zu entries, got %zu\n", name, expected, actual);
    throw std::runtime_error(std::string("Distributed size mismatch: ") + name);
}

float reportVectorDifference(
    const char *name,
    const std::vector<float> &expected,
    const std::vector<float> &actual) {
    requireSize(name, expected.size(), actual.size());
    float maximum = 0.0f;
    double squared = 0.0;
    for (std::size_t i = 0u; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]))
            throw std::runtime_error(std::string("Non-finite distributed value: ") + name);
        const float difference = std::fabs(expected[i] - actual[i]);
        maximum = std::max(maximum, difference);
        squared += static_cast<double>(difference) * difference;
    }
    const double rms = std::sqrt(squared / expected.size());
    std::fprintf(stderr, "%s: max_abs=%.9g rms=%.9g\n", name, maximum, rms);
    return maximum;
}

float vectorRmsDifference(
    const char *name,
    const std::vector<float> &expected,
    const std::vector<float> &actual) {
    requireSize(name, expected.size(), actual.size());
    double squared = 0.0;
    for (std::size_t i = 0u; i < expected.size(); ++i) {
        if (!std::isfinite(expected[i]) || !std::isfinite(actual[i]))
            throw std::runtime_error(std::string("Non-finite distributed value: ") + name);
        const double difference = static_cast<double>(expected[i]) - actual[i];
        squared += difference * difference;
    }
    const float rms = static_cast<float>(std::sqrt(squared / expected.size()));
    std::fprintf(stderr, "%s: rms=%.9g\n", name, rms);
    return rms;
}

float reportLogitDifference(const DistributedResult &expected, const DistributedResult &actual) {
    if (expected.firstPassLogits.size() % expected.localVocab != 0u ||
        actual.firstPassLogits.size() % actual.localVocab != 0u)
        throw std::runtime_error("Invalid distributed-test logit dimensions");
    const std::size_t positions = expected.firstPassLogits.size() / expected.localVocab;
    requireSize("logit positions", positions, actual.firstPassLogits.size() / actual.localVocab);
    float maximum = 0.0f;
    double squared = 0.0;
    std::size_t count = 0u;
    for (std::size_t position = 0u; position < positions; ++position) {
        for (std::size_t token = 0u; token < actual.localVocab; ++token) {
            const float baseline = expected.firstPassLogits[position * expected.localVocab + token];
            const float current = actual.firstPassLogits[position * actual.localVocab + token];
            if (!std::isfinite(baseline) || !std::isfinite(current))
                throw std::runtime_error("Non-finite distributed value: first-pass root logits");
            const float difference = std::fabs(baseline - current);
            maximum = std::max(maximum, difference);
            squared += static_cast<double>(difference) * difference;
            ++count;
        }
    }
    std::fprintf(stderr, "first-pass root logits: max_abs=%.9g rms=%.9g\n",
        maximum, std::sqrt(squared / count));
    return maximum;
}

float logitRmsDifference(const DistributedResult &expected, const DistributedResult &actual) {
    if (expected.firstPassLogits.size() % expected.localVocab != 0u ||
        actual.firstPassLogits.size() % actual.localVocab != 0u)
        throw std::runtime_error("Invalid distributed-test logit dimensions");
    const std::size_t positions = expected.firstPassLogits.size() / expected.localVocab;
    requireSize("logit positions", positions, actual.firstPassLogits.size() / actual.localVocab);
    double squared = 0.0;
    std::size_t count = 0u;
    for (std::size_t position = 0u; position < positions; ++position) {
        for (std::size_t token = 0u; token < actual.localVocab; ++token) {
            const float baseline = expected.firstPassLogits[position * expected.localVocab + token];
            const float current = actual.firstPassLogits[position * actual.localVocab + token];
            if (!std::isfinite(baseline) || !std::isfinite(current))
                throw std::runtime_error("Non-finite distributed value: first-pass root logits");
            const double difference = static_cast<double>(baseline) - current;
            squared += difference * difference;
            ++count;
        }
    }
    const float rms = static_cast<float>(std::sqrt(squared / count));
    std::fprintf(stderr, "first-pass root logits RMS: rms=%.9g\n", rms);
    return rms;
}

float reportRowSliceDifference(
    const char *name,
    const std::vector<float> &expected,
    std::size_t expectedWidth,
    const std::vector<float> &actual,
    std::size_t actualWidth,
    std::size_t expectedStart) {
    if (expected.size() % expectedWidth != 0u || actual.size() % actualWidth != 0u)
        throw std::runtime_error(std::string("Invalid row dimensions: ") + name);
    const std::size_t rows = expected.size() / expectedWidth;
    requireSize(name, rows, actual.size() / actualWidth);
    if (expectedStart > expectedWidth || actualWidth > expectedWidth - expectedStart)
        throw std::runtime_error(std::string("Actual row is outside the baseline: ") + name);
    float maximum = 0.0f;
    double squared = 0.0;
    for (std::size_t row = 0u; row < rows; ++row) {
        for (std::size_t column = 0u; column < actualWidth; ++column) {
            const float baseline = expected[row * expectedWidth + expectedStart + column];
            const float current = actual[row * actualWidth + column];
            if (!std::isfinite(baseline) || !std::isfinite(current))
                throw std::runtime_error(std::string("Non-finite distributed value: ") + name);
            const float difference = std::fabs(baseline - current);
            maximum = std::max(maximum, difference);
            squared += static_cast<double>(difference) * difference;
        }
    }
    std::fprintf(stderr, "%s: max_abs=%.9g rms=%.9g\n", name, maximum,
        std::sqrt(squared / (rows * actualWidth)));
    return maximum;
}

void requireMaximum(const char *name, float maximum, float tolerance) {
    if (maximum <= tolerance)
        return;
    std::fprintf(stderr, "%s maximum difference %.9g exceeds %.9g\n", name, maximum, tolerance);
    throw std::runtime_error(std::string("Distributed tensor mismatch: ") + name);
}

std::vector<float> q80ContributionSum(const std::vector<NodeSnapshot> &nodes) {
    if (nodes.empty())
        throw std::runtime_error("Cannot sum an empty set of Q80 contributions");
    const std::size_t size = nodes[0].attentionContribution.size();
    if (size % Q80_BLOCK_SIZE != 0u)
        throw std::runtime_error("Q80 contribution size is not block aligned");
    std::vector<float> sum(size, 0.0f);
    std::vector<float> unpacked(size);
    std::vector<NnBlockQ80> packed(size / Q80_BLOCK_SIZE);
    for (std::size_t node = 0u; node < nodes.size(); ++node) {
        requireSize("Q80 attention contribution", size,
            nodes[node].attentionContribution.size());
        quantizeF32toQ80(nodes[node].attentionContribution.data(), packed.data(),
            static_cast<NnUint>(size), 1u, 0u);
        dequantizeQ80toF32(packed.data(), unpacked.data(),
            static_cast<NnUint>(size), 1u, 0u);
        for (std::size_t i = 0u; i < size; ++i)
            sum[i] += unpacked[i];
    }
    return sum;
}

void reportNodeAttentionDifference(const DistributedResult &expected, const DistributedResult &actual) {
    requireSize("node snapshots", 1u, expected.nodes.size());
    if (actual.nodes.empty())
        throw std::runtime_error("Distributed execution produced no node snapshots");
    const NodeSnapshot &baseline = expected.nodes[0];
    if (baseline.queryNormWeight.empty() || baseline.kvWidth % baseline.queryNormWeight.size() != 0u)
        throw std::runtime_error("Invalid baseline KV-head dimensions");
    const std::size_t kvHeads = baseline.kvWidth / baseline.queryNormWeight.size();
    const std::size_t kvReplication = std::max<std::size_t>(1u, actual.nodes.size() / kvHeads);
    std::vector<float> contributionSum(baseline.attentionContribution.size(), 0.0f);
    for (std::size_t node = 0u; node < actual.nodes.size(); ++node) {
        const NodeSnapshot &current = actual.nodes[node];
        const std::string prefix = "node " + std::to_string(node) + " ";
        requireMaximum("tokens", reportVectorDifference((prefix + "tokens").c_str(),
            baseline.tokens, current.tokens), 0.0f);
        requireMaximum("positions", reportVectorDifference((prefix + "positions").c_str(),
            baseline.positions, current.positions), 0.0f);
        requireMaximum("query norm weight", reportVectorDifference((prefix + "query norm weight").c_str(),
            baseline.queryNormWeight, current.queryNormWeight), 0.0f);
        requireMaximum("key norm weight", reportVectorDifference((prefix + "key norm weight").c_str(),
            baseline.keyNormWeight, current.keyNormWeight), 0.0f);
        requireMaximum("rope cache", reportVectorDifference((prefix + "rope cache").c_str(),
            baseline.ropeCache, current.ropeCache), 0.0f);
        requireMaximum("query", reportRowSliceDifference((prefix + "query").c_str(),
            baseline.query, baseline.queryWidth, current.query, current.queryWidth,
            node * current.queryWidth), 0.0f);
        const std::size_t kvStart = (node / kvReplication) * current.kvWidth;
        requireMaximum("key", reportRowSliceDifference((prefix + "key").c_str(),
            baseline.key, baseline.kvWidth, current.key, current.kvWidth, kvStart), 0.0f);
        requireMaximum("value", reportRowSliceDifference((prefix + "value").c_str(),
            baseline.value, baseline.kvWidth, current.value, current.kvWidth, kvStart), 0.0f);
        requireMaximum("attention", reportRowSliceDifference((prefix + "attention").c_str(),
            baseline.attention, baseline.queryWidth, current.attention, current.queryWidth,
            node * current.queryWidth), 0.0f);
        requireSize("attention contribution", contributionSum.size(), current.attentionContribution.size());
        for (std::size_t i = 0u; i < contributionSum.size(); ++i)
            contributionSum[i] += current.attentionContribution[i];
    }
    const float baselineContributionDifference = reportVectorDifference(
        "offline attention contribution sum", baseline.attentionContribution, contributionSum);
    // For Q80 synchronization, every layer after the first consumes a deliberately
    // quantized reduction. Multi-node execution quantizes each shard's partial before
    // summation, whereas one-node execution quantizes the complete contribution once,
    // so their final-layer local contributions need not match. One-layer Q40 and all
    // F32 runs retain the stricter cross-run assertion.
    if (!actual.q80Sync || actual.layerCount == 1u)
        requireMaximum("offline attention contribution sum", baselineContributionDifference, 5e-5f);
    const std::vector<float> synchronizedReference = actual.q80Sync
        ? q80ContributionSum(actual.nodes)
        : contributionSum;
    requireMaximum("synchronized attention projection",
        reportVectorDifference("synchronized attention projection",
            synchronizedReference, actual.nodes[0].attentionProjection), 1e-6f);
}

void assertEquivalent(const DistributedResult &expected, const DistributedResult &actual) {
    if (expected.q80Sync != actual.q80Sync)
        throw std::runtime_error("Cannot compare different distributed synchronization formats");
    if (expected.layerCount != actual.layerCount)
        throw std::runtime_error("Cannot compare different distributed layer counts");
    const bool q80Sync = actual.q80Sync;
    const float routerTolerance = q80Sync ? 5e-3f : 1e-6f;
    const float expertWeightTolerance = q80Sync ? 1e-2f : 1e-6f;
    const float branchTolerance = q80Sync ? 100.0f : 1e-3f;
    const float hiddenTolerance = q80Sync ? 500.0f : 5e-3f;
    const float logitTolerance = q80Sync ? 60.0f : 2e-3f;
    // Each transformer layer introduces two Q80 synchronization boundaries.
    // Keep the measured one-layer allowance as the per-layer budget so compact
    // multi-layer fixtures test bounded accumulation instead of incorrectly
    // imposing the one-layer total on the entire graph.
    const float statisticTolerance = q80Sync ? 0.1f * actual.layerCount : 1e-5f;
    // Tail-token log probabilities amplify bounded logit changes. Keep the
    // strict one-layer qualification unchanged, but give multi-layer Q80 runs
    // a separate per-layer budget without relaxing entropy comparisons.
    const float logProbabilityTolerance = q80Sync && actual.layerCount > 1u
        ? 2.5f * actual.layerCount
        : statisticTolerance;
    requireSize("scores", expected.scores.size(), actual.scores.size());
    requireSize("samples", expected.samples.size(), actual.samples.size());
    requireSize("conditioned samples", expected.conditionedSamples.size(), actual.conditionedSamples.size());
    if (q80Sync) {
        std::size_t scoreArgmaxMismatches = 0u;
        std::size_t sampleTokenMismatches = 0u;
        std::size_t sampleArgmaxMismatches = 0u;
        std::size_t conditionedTokenMismatches = 0u;
        std::size_t conditionedArgmaxMismatches = 0u;
        for (std::size_t i = 0u; i < expected.scores.size(); ++i)
            scoreArgmaxMismatches += expected.scores[i].argmaxToken != actual.scores[i].argmaxToken;
        for (std::size_t i = 0u; i < expected.samples.size(); ++i) {
            sampleTokenMismatches += expected.samples[i].token != actual.samples[i].token;
            sampleArgmaxMismatches += expected.samples[i].argmaxToken != actual.samples[i].argmaxToken;
        }
        for (std::size_t i = 0u; i < expected.conditionedSamples.size(); ++i) {
            conditionedTokenMismatches +=
                expected.conditionedSamples[i].token != actual.conditionedSamples[i].token;
            conditionedArgmaxMismatches +=
                expected.conditionedSamples[i].argmaxToken != actual.conditionedSamples[i].argmaxToken;
        }
        std::fprintf(stderr,
            "Q80 discrete drift: score_argmax=%zu sample_token=%zu sample_argmax=%zu "
            "conditioned_token=%zu conditioned_argmax=%zu\n",
            scoreArgmaxMismatches, sampleTokenMismatches, sampleArgmaxMismatches,
            conditionedTokenMismatches, conditionedArgmaxMismatches);
    }
    requireMaximum("hard embedding",
        reportVectorDifference("hard embedding", expected.hardEmbedding, actual.hardEmbedding), 0.0f);
    reportNodeAttentionDifference(expected, actual);
    requireMaximum("router scores",
        reportVectorDifference("router scores", expected.routerScores, actual.routerScores), routerTolerance);
    const float expertIndexDifference = reportVectorDifference(
        "expert indexes", expected.expertIndexes, actual.expertIndexes);
    if (!q80Sync)
        requireMaximum("expert indexes", expertIndexDifference, 0.0f);
    requireMaximum("expert weights",
        reportVectorDifference("expert weights", expected.expertWeights, actual.expertWeights),
        expertWeightTolerance);
    requireMaximum("normalized branches",
        reportVectorDifference("normalized branches", expected.branches, actual.branches), branchTolerance);
    requireMaximum("combined branches",
        reportVectorDifference("combined branches", expected.combinedBranches, actual.combinedBranches),
        branchTolerance);
    requireMaximum("first-pass hidden",
        reportVectorDifference("first-pass hidden", expected.firstPassHidden, actual.firstPassHidden),
        hiddenTolerance);
    requireMaximum("first-pass root logits", reportLogitDifference(expected, actual), logitTolerance);
    if (q80Sync) {
        requireMaximum("normalized branches RMS",
            vectorRmsDifference("normalized branches RMS", expected.branches, actual.branches), 2.0f);
        requireMaximum("combined branches RMS",
            vectorRmsDifference("combined branches RMS", expected.combinedBranches,
                actual.combinedBranches), 2.0f);
        requireMaximum("first-pass hidden RMS",
            vectorRmsDifference("first-pass hidden RMS", expected.firstPassHidden,
                actual.firstPassHidden), 20.0f);
        requireMaximum("first-pass root logits RMS", logitRmsDifference(expected, actual), 10.0f);
    }
    for (std::size_t i = 0u; i < expected.scores.size(); ++i) {
        if (!q80Sync)
            requireEqual("score argmax", i,
                expected.scores[i].argmaxToken, actual.scores[i].argmaxToken);
        requireNear("score probability", i,
            expected.scores[i].probability, actual.scores[i].probability,
            q80Sync ? 1e-4f : 1e-7f);
        requireNear("score log probability", i,
            expected.scores[i].logProbability, actual.scores[i].logProbability,
            logProbabilityTolerance);
        requireNear("score entropy", i,
            expected.scores[i].entropy, actual.scores[i].entropy, statisticTolerance);
    }
    for (std::size_t i = 0u; i < expected.samples.size(); ++i) {
        if (!q80Sync) {
            requireEqual("sample token", i, expected.samples[i].token, actual.samples[i].token);
            requireEqual("sample argmax", i,
                expected.samples[i].argmaxToken, actual.samples[i].argmaxToken);
        }
        requireNear("sample entropy", i,
            expected.samples[i].entropy, actual.samples[i].entropy, statisticTolerance);
    }
    for (std::size_t i = 0u; i < expected.conditionedSamples.size(); ++i) {
        if (!q80Sync) {
            requireEqual("conditioned sample token", i,
                expected.conditionedSamples[i].token, actual.conditionedSamples[i].token);
            requireEqual("conditioned sample argmax", i,
                expected.conditionedSamples[i].argmaxToken, actual.conditionedSamples[i].argmaxToken);
        }
        requireNear("conditioned sample entropy", i,
            expected.conditionedSamples[i].entropy, actual.conditionedSamples[i].entropy,
            q80Sync ? statisticTolerance : 5e-5f);
    }
}

void testFormat(const char *path, std::uint32_t portBase) {
    std::fprintf(stderr, "distributed test: %s, 1 node\n", path);
    const DistributedResult baseline = runModel(path, 1u, portBase);
    const NnUint nodeCounts[] = {2u, 4u, 8u};
    for (std::size_t i = 0u; i < sizeof(nodeCounts) / sizeof(nodeCounts[0]); ++i) {
        std::fprintf(stderr, "distributed test: %s, %u nodes\n", path, nodeCounts[i]);
        assertEquivalent(baseline, runModel(path, nodeCounts[i], portBase + 16u * (i + 1u)));
    }
}

void testCheckpointFormat(const char *path, std::uint32_t portBase) {
    const std::uint32_t queryLength = 3u;
    const std::uint32_t context = queryLength * 2u;
    std::fprintf(stderr, "checkpoint distributed test: %s, 1 node\n", path);
    const DistributedResult baseline = runModel(path, 1u, portBase, queryLength, context);
    const NnUint nodeCounts[] = {2u, 4u, 8u};
    for (std::size_t i = 0u; i < sizeof(nodeCounts) / sizeof(nodeCounts[0]); ++i) {
        std::fprintf(stderr, "checkpoint distributed test: %s, %u nodes\n", path, nodeCounts[i]);
        assertEquivalent(baseline,
            runModel(path, nodeCounts[i], portBase + 16u * (i + 1u), queryLength, context));
    }
}

void testCheckpointNodes(const char *path, NnUint nNodes, std::uint32_t portBase) {
    const std::uint32_t queryLength = 3u;
    const std::uint32_t context = queryLength * 2u;
    const DistributedResult baseline = runModel(path, 1u, portBase, queryLength, context);
    assertEquivalent(baseline, runModel(path, nNodes, portBase + 16u, queryLength, context));
}

} // namespace

int main(int argc, char **argv) {
    assert(argc >= 2);
    initSockets();
    initQuants();
    const std::uint32_t processOffset = static_cast<std::uint32_t>(DDLLM_GETPID()) % 1000u;
    if (argc == 4 && std::string(argv[1]) == "--checkpoint-nodes") {
        const NnUint nNodes = static_cast<NnUint>(std::strtoul(argv[2], NULL, 10));
        testCheckpointNodes(argv[3], nNodes, 20000u + processOffset * 16u);
        cleanupSockets();
        return 0;
    }
    if (argc == 3 && std::string(argv[1]) == "--checkpoint") {
        testCheckpointFormat(argv[2], 20000u + processOffset * 16u);
        cleanupSockets();
        return 0;
    }
    if (argc == 4 && std::string(argv[1]) == "--nodes") {
        const NnUint nNodes = static_cast<NnUint>(std::strtoul(argv[2], NULL, 10));
        const DistributedResult baseline = runModel(argv[3], 1u, 20000u + processOffset * 16u);
        assertEquivalent(baseline, runModel(argv[3], nNodes, 30000u + processOffset * 16u));
        cleanupSockets();
        return 0;
    }
    for (int i = 1; i < argc; ++i)
        testFormat(argv[i], 20000u + processOffset * 16u + static_cast<std::uint32_t>(i - 1) * 20000u);
    cleanupSockets();
    return 0;
}
