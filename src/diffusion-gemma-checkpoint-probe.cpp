#include "diffusion-gemma-loader.hpp"
#include "diffusion-gemma-runtime.hpp"
#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

const std::uint32_t TRACE_MAGIC = UINT32_C(0x46524444); // DDRF in little-endian byte order
const std::uint32_t TRACE_VERSION = 1u;
const NnUint QUERY_LENGTH = 3u;
const NnUint CONTEXT_CAPACITY = 8u;
const NnUint CACHE_FIXTURE_LENGTH = 6u;
const NnUint CACHE_FIXTURE_CAPACITY = 4u;

class TraceWriter {
private:
    std::fstream output_;
    std::uint32_t recordCount_;

    template <typename T>
    void writeValue(const T &value) {
        output_.write(reinterpret_cast<const char *>(&value), sizeof(value));
        if (!output_)
            throw std::runtime_error("Failed to write checkpoint trace");
    }

public:
    explicit TraceWriter(const char *path) : output_(path, std::ios::binary | std::ios::out | std::ios::trunc), recordCount_(0u) {
        if (!output_)
            throw std::runtime_error(std::string("Cannot create checkpoint trace: ") + path);
        writeValue(TRACE_MAGIC);
        writeValue(TRACE_VERSION);
        writeValue(recordCount_);
    }

    void add(const std::string &name, const std::vector<std::uint32_t> &shape, const float *values) {
        if (name.empty() || values == NULL || shape.empty())
            throw std::invalid_argument("Invalid checkpoint trace record");
        std::uint64_t count = 1u;
        for (std::size_t i = 0u; i < shape.size(); ++i) {
            if (shape[i] == 0u || count > UINT64_MAX / shape[i])
                throw std::invalid_argument("Invalid checkpoint trace shape");
            count *= shape[i];
        }
        const std::uint32_t nameLength = static_cast<std::uint32_t>(name.size());
        const std::uint32_t rank = static_cast<std::uint32_t>(shape.size());
        writeValue(nameLength);
        writeValue(rank);
        writeValue(count);
        output_.write(name.data(), name.size());
        output_.write(reinterpret_cast<const char *>(shape.data()), shape.size() * sizeof(shape[0]));
        output_.write(reinterpret_cast<const char *>(values), count * sizeof(float));
        if (!output_)
            throw std::runtime_error("Failed to write checkpoint trace payload");
        ++recordCount_;
    }

    void add(const std::string &name, const std::vector<std::uint32_t> &shape, const std::vector<float> &values) {
        std::uint64_t expected = 1u;
        for (std::size_t i = 0u; i < shape.size(); ++i)
            expected *= shape[i];
        if (expected != values.size())
            throw std::invalid_argument("Checkpoint trace vector size does not match its shape");
        add(name, shape, values.data());
    }

    void finish() {
        output_.seekp(2u * sizeof(std::uint32_t));
        writeValue(recordCount_);
        output_.flush();
        if (!output_)
            throw std::runtime_error("Failed to finalize checkpoint trace");
    }
};

NnUint bufferIndex(const NnNodeConfig &node, const std::string &name) {
    for (NnUint i = 0u; i < node.nBuffers; ++i)
        if (name == node.buffers[i].name)
            return i;
    throw std::runtime_error("Missing checkpoint probe buffer: " + name);
}

void addRuntimeRecords(
    TraceWriter *writer,
    const std::string &prefix,
    const DiffusionGemmaHeader &header,
    const DiffusionGemmaGraph &graph,
    const NnCpuDevice *cpu,
    const NnNetExecution &execution,
    NnUint cacheLength) {
    const NnNodeConfig &node = graph.nodes[0];
    const NnUint headDim = header.layerTypes[0] == DIFFUSION_GEMMA_SLIDING_ATTENTION
        ? header.slidingHeadDim : header.fullHeadDim;
    const NnUint kvHeads = header.layerTypes[0] == DIFFUSION_GEMMA_SLIDING_ATTENTION
        ? header.slidingKvHeads : header.fullKvHeads;
    const NnUint queryDim = header.queryHeads * headDim;
    const NnUint kvDim = kvHeads * headDim;
    const std::vector<std::uint32_t> hiddenShape{QUERY_LENGTH, header.hiddenSize};
    const std::vector<std::uint32_t> queryShape{QUERY_LENGTH, queryDim};
    const std::vector<std::uint32_t> kvShape{QUERY_LENGTH, kvDim};

    writer->add(prefix + ".hard_embedding", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "hard_embedding")]));
    writer->add(prefix + ".hidden", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "hidden")]));
    writer->add(prefix + ".query", queryShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.query")]));
    writer->add(prefix + ".key", kvShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.key")]));
    writer->add(prefix + ".value", kvShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.value")]));
    writer->add(prefix + ".attention", queryShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.attention")]));
    writer->add(prefix + ".attention_projection", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "local_attention")]));
    writer->add(prefix + ".router_probabilities", {QUERY_LENGTH, header.expertCount},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "router_scores")]));
    writer->add(prefix + ".expert_indexes", {QUERY_LENGTH, header.activeExpertCount},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "expert_indexes")]));
    writer->add(prefix + ".expert_weights", {header.activeExpertCount, QUERY_LENGTH},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "expert_weights")]));
    writer->add(prefix + ".dense_output", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "local_dense")]));
    writer->add(prefix + ".moe_output", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "local_moe")]));
    writer->add(prefix + ".normalized_branches", {2u, QUERY_LENGTH, header.hiddenSize},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "branches")]));
    writer->add(prefix + ".combined_branches", hiddenShape,
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "branch_combined")]));
    writer->add(prefix + ".key_cache", {cacheLength, kvDim},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.key_cache")]));
    writer->add(prefix + ".value_cache", {cacheLength, kvDim},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.value_cache")]));
    writer->add(prefix + ".logits", {QUERY_LENGTH, header.vocabSize},
        reinterpret_cast<const float *>(execution.pipes[graph.pipes.localLogits]));
    writer->add(prefix + ".self_conditioning", hiddenShape,
        reinterpret_cast<const float *>(execution.pipes[graph.pipes.selfConditioning]));
}

std::vector<float> sampleRecords(const std::vector<DiffusionGemmaSampledPosition> &samples) {
    std::vector<float> values(samples.size() * 3u);
    for (std::size_t i = 0u; i < samples.size(); ++i) {
        values[i * 3u] = static_cast<float>(samples[i].token);
        values[i * 3u + 1u] = static_cast<float>(samples[i].argmaxToken);
        values[i * 3u + 2u] = samples[i].entropy;
    }
    return values;
}

void run(const char *modelPath, const char *outputPath, NnUint nThreads) {
    initQuants();
    DiffusionGemmaModelFile model(modelPath);
    const DiffusionGemmaHeader &header = model.header();
    if (header.layerCount != 1u)
        throw std::runtime_error("Checkpoint reference traces require exactly one converted layer");
    if (header.vocabSize <= 42u || header.maxPositionEmbeddings < CONTEXT_CAPACITY)
        throw std::runtime_error("Checkpoint model is too small for the fixed reference fixture");

    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(header, 1u, QUERY_LENGTH, CONTEXT_CAPACITY);
    std::unique_ptr<DiffusionGemmaGraph, void (*)(DiffusionGemmaGraph *)> graphGuard(
        &graph, releaseDiffusionGemmaGraph);
    NnNetExecution execution(nThreads, &graph.net);
    std::vector<NnExecutorDevice> devices;
    NnCpuDevice *cpu = new NnCpuDevice(&graph.net, &graph.nodes[0], &execution);
    devices.push_back(NnExecutorDevice(cpu, -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, &synchronizer, false);
    NnRootWeightLoader weightLoader(&executor, NULL, 1u);
    loadDiffusionGemmaWeights(model, graph, &weightLoader);
    RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, NULL);
    TraceWriter writer(outputPath);

    const std::vector<std::int32_t> encoderTokens{2, 17, 42};
    const std::vector<std::int32_t> canvasTokens{7, 23, 31};
    const std::vector<double> coins{0.125, 0.5, 0.875};
    std::vector<float> encoderTokenValues(encoderTokens.begin(), encoderTokens.end());
    std::vector<float> canvasTokenValues(canvasTokens.begin(), canvasTokens.end());
    std::vector<float> coinValues(coins.begin(), coins.end());
    writer.add("fixture.encoder_tokens", {QUERY_LENGTH}, encoderTokenValues);
    writer.add("fixture.canvas_tokens", {QUERY_LENGTH}, canvasTokenValues);
    writer.add("fixture.sampling_coins", {QUERY_LENGTH}, coinValues);
    writer.add("fixture.layer_type", {1u}, std::vector<float>{
        static_cast<float>(header.layerTypes[0])});
    writer.add("fixture.weight_type", {1u}, std::vector<float>{
        static_cast<float>(header.weightType)});

    runtime.resetSelfConditioning();
    runtime.encode(encoderTokens, 0u, 0u, true);
    addRuntimeRecords(&writer, "encoder", header, graph, cpu, execution, QUERY_LENGTH);

    const std::vector<DiffusionGemmaSampledPosition> decoder0 = runtime.denoise(
        canvasTokens, QUERY_LENGTH, QUERY_LENGTH, 0.8f, coins, 0u, 2u, false);
    addRuntimeRecords(&writer, "decoder0", header, graph, cpu, execution, QUERY_LENGTH);
    writer.add("decoder0.samples", {QUERY_LENGTH, 3u}, sampleRecords(decoder0));

    const std::vector<DiffusionGemmaSampledPosition> decoder1 = runtime.denoise(
        canvasTokens, QUERY_LENGTH, QUERY_LENGTH, 0.8f, coins, 0u, 1u, true);
    addRuntimeRecords(&writer, "decoder1", header, graph, cpu, execution, QUERY_LENGTH);
    writer.add("decoder1.samples", {QUERY_LENGTH, 3u}, sampleRecords(decoder1));
    writer.finish();
}

void runEncoderCache(
    const char *modelPath,
    const char *outputPath,
    NnUint chunkSize,
    NnUint nThreads) {
    initQuants();
    DiffusionGemmaModelFile model(modelPath);
    const DiffusionGemmaHeader &header = model.header();
    if (header.layerCount != 1u)
        throw std::runtime_error("Encoder cache comparison requires exactly one converted layer");
    if (header.layerTypes[0] != DIFFUSION_GEMMA_SLIDING_ATTENTION)
        throw std::runtime_error("Encoder cache comparison requires a sliding-attention layer");
    if (header.vocabSize <= 42u || header.maxPositionEmbeddings < CONTEXT_CAPACITY)
        throw std::runtime_error("Checkpoint model is too small for the cache fixture");
    if (chunkSize == 0u || chunkSize > 3u ||
        CACHE_FIXTURE_LENGTH % chunkSize != 0u)
        throw std::invalid_argument("Encoder cache chunk must be one of 1, 2, or 3");

    DiffusionGemmaHeader fixtureHeader = header;
    fixtureHeader.slidingWindow = CACHE_FIXTURE_CAPACITY;
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(
        fixtureHeader, 1u, chunkSize, CONTEXT_CAPACITY);
    std::unique_ptr<DiffusionGemmaGraph, void (*)(DiffusionGemmaGraph *)> graphGuard(
        &graph, releaseDiffusionGemmaGraph);
    NnNetExecution execution(nThreads, &graph.net);
    std::vector<NnExecutorDevice> devices;
    NnCpuDevice *cpu = new NnCpuDevice(&graph.net, &graph.nodes[0], &execution);
    devices.push_back(NnExecutorDevice(cpu, -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, &synchronizer, false);
    NnRootWeightLoader weightLoader(&executor, NULL, 1u);
    loadDiffusionGemmaWeights(model, graph, &weightLoader);
    RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, NULL);

    const std::vector<std::int32_t> tokens{2, 17, 42, 7, 23, 31};
    for (NnUint start = 0u; start < CACHE_FIXTURE_LENGTH; start += chunkSize) {
        const std::vector<std::int32_t> chunk(
            tokens.begin() + start, tokens.begin() + start + chunkSize);
        runtime.encode(chunk, start, start, false);
    }

    const NnUint headDim = header.layerTypes[0] == DIFFUSION_GEMMA_SLIDING_ATTENTION
        ? header.slidingHeadDim : header.fullHeadDim;
    const NnUint kvHeads = header.layerTypes[0] == DIFFUSION_GEMMA_SLIDING_ATTENTION
        ? header.slidingKvHeads : header.fullKvHeads;
    const NnUint kvDim = kvHeads * headDim;
    const NnNodeConfig &node = graph.nodes[0];
    TraceWriter writer(outputPath);
    writer.add("fixture.encoder_tokens", {CACHE_FIXTURE_LENGTH},
        std::vector<float>(tokens.begin(), tokens.end()));
    writer.add("fixture.layer_type", {1u}, std::vector<float>{
        static_cast<float>(header.layerTypes[0])});
    writer.add("fixture.weight_type", {1u}, std::vector<float>{
        static_cast<float>(header.weightType)});
    writer.add("fixture.encoder_chunk", {1u}, std::vector<float>{
        static_cast<float>(chunkSize)});
    writer.add("fixture.cache_capacity", {1u}, std::vector<float>{
        static_cast<float>(CACHE_FIXTURE_CAPACITY)});
    writer.add("encoder.key_cache", {CACHE_FIXTURE_CAPACITY, kvDim},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.key_cache")]));
    writer.add("encoder.value_cache", {CACHE_FIXTURE_CAPACITY, kvDim},
        reinterpret_cast<const float *>(cpu->buffers[bufferIndex(node, "layer.0.value_cache")]));
    writer.finish();
}

NnUint parseThreads(const char *value) {
    char *end = NULL;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (value == end || *end != '\0' || parsed == 0u || parsed > UINT32_MAX)
        throw std::invalid_argument("Invalid checkpoint probe thread count");
    return static_cast<NnUint>(parsed);
}

} // namespace

int main(int argc, char **argv) {
    const bool cacheMode = argc >= 5 && std::strcmp(argv[3], "--encoder-cache") == 0;
    if ((!cacheMode && (argc < 3 || argc > 4)) || (cacheMode && argc > 6)) {
        std::fprintf(stderr,
            "Usage: %s MODEL.ddlm OUTPUT.ddrf [NTHREADS]\n"
            "       %s MODEL.ddlm OUTPUT.ddrf --encoder-cache CHUNK [NTHREADS]\n",
            argv[0], argv[0]);
        return 1;
    }
    try {
        if (cacheMode) {
            const NnUint chunkSize = parseThreads(argv[4]);
            runEncoderCache(argv[1], argv[2], chunkSize,
                argc == 6 ? parseThreads(argv[5]) : 1u);
        } else {
            run(argv[1], argv[2], argc == 4 ? parseThreads(argv[3]) : 1u);
        }
        std::printf("Checkpoint trace written to %s\n", argv[2]);
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "error: %s\n", error.what());
        return 2;
    }
}
