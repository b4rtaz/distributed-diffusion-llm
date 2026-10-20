// Ordinary one-layer Q40 CPU decoder benchmark, with optional identical-input
// Vulkan head replay. The replay isolates tied embedding from upstream routing.
#include "../diffusion-gemma-loader.hpp"
#include "../diffusion-gemma-runtime.hpp"
#include "nn-cpu.hpp"
#include "nn-network.hpp"
#ifdef DLLAMA_VULKAN
#include "nn-vulkan.hpp"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace {
struct Snapshot {
    std::vector<float> logits;
    std::vector<float> mix;
    std::vector<DiffusionGemmaSampledPosition> samples;
    std::vector<NnByte> quantHidden;
    std::vector<float> normalizedMix;
};

void compare(const char *name, const float *values, const std::vector<float> &reference,
    double maxLimit, double rmsLimit) {
    double maximum = 0.0, squared = 0.0;
    for (std::size_t i = 0u; i < reference.size(); ++i) {
        if (!std::isfinite(values[i]) || !std::isfinite(reference[i]))
            throw std::runtime_error(std::string("non-finite ") + name);
        const double error = std::fabs(static_cast<double>(values[i]) - reference[i]);
        maximum = std::max(maximum, error);
        squared += error * error;
    }
    const double rms = std::sqrt(squared / reference.size());
    std::printf("  %s: max %.9g, RMS %.9g\n", name, maximum, rms);
    if (maximum > maxLimit || rms > rmsLimit)
        throw std::runtime_error(std::string(name) + " exceeds Q40 comparison bounds");
}

void run(const DiffusionGemmaModelFile &model, NnUint threads, NnUint batches,
    NnUint repeats, bool vulkan, bool replayHead, Snapshot (&reference)[2]) {
    const auto &h = model.header();
    auto graph = buildDiffusionGemmaGraph(h, 1u, std::max(3u, batches), batches + 3u);
    std::unique_ptr<DiffusionGemmaGraph, void (*)(DiffusionGemmaGraph *)> guard(
        &graph, releaseDiffusionGemmaGraph);
    NnNetExecution execution(vulkan ? 1u : threads, &graph.net);
    std::vector<NnExecutorDevice> devices;
#ifdef DLLAMA_VULKAN
    if (vulkan)
        devices.emplace_back(new NnVulkanDevice(0u, &graph.net, &graph.nodes[0], &execution), -1, -1);
    else
#endif
        devices.emplace_back(new NnCpuDevice(&graph.net, &graph.nodes[0], &execution), -1, -1);
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, &synchronizer, true);
    NnUint quantIndex = UINT32_MAX, mixIndex = UINT32_MAX, headSegment = UINT32_MAX;
    for (NnUint i = 0u; i < graph.nodes[0].nBuffers; ++i) {
        if (std::strcmp(graph.nodes[0].buffers[i].name, "quant_hidden") == 0)
            quantIndex = i;
        if (std::strcmp(graph.nodes[0].buffers[i].name, "local_embedding_mix") == 0)
            mixIndex = i;
    }
    for (NnUint s = graph.decoderDenoise.first; s <= graph.decoderDenoise.last; ++s) {
        const auto &segment = graph.nodes[0].segments[s];
        for (NnUint op = 0u; op < segment.nOps; ++op)
            if (segment.ops[op].code == OP_TIED_EMBEDDING_PARTIAL)
                headSegment = s;
    }
    if (quantIndex == UINT32_MAX || mixIndex == UINT32_MAX || headSegment == UINT32_MAX)
        throw std::runtime_error("missing tied-embedding graph buffers or segment");
    NnRootWeightLoader loader(&executor, NULL, 1u);
    loadDiffusionGemmaWeights(model, graph, &loader);
    RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, NULL);
    if (!replayHead)
        runtime.encode({2, 17, 42}, 0u, 0u);
    std::vector<std::int32_t> canvas(batches);
    std::vector<double> coins(batches);
    for (NnUint b = 0u; b < batches; ++b) {
        canvas[b] = (7u + b * 1009u) % h.vocabSize;
        coins[b] = (b + 0.5) / batches;
    }
    std::printf("%s: %u layer(s), %u positions, %u executor thread(s)\n",
        replayHead ? "Vulkan identical-input head replay" : vulkan ? "Vulkan full decoder" : "CPU",
        h.layerCount, batches, execution.nThreads);
    for (NnUint repeat = 0u; repeat < repeats; ++repeat) {
        runtime.resetSelfConditioning();
        for (NnUint pass = 0u; pass < 2u; ++pass) {
            std::vector<DiffusionGemmaSampledPosition> samples;
            if (replayHead) {
                execution.setBatchSize(batches);
                reinterpret_cast<float *>(execution.pipes[graph.pipes.temperature])[0] = 0.8f;
                for (NnUint b = 0u; b < batches; ++b)
                    reinterpret_cast<float *>(execution.pipes[graph.pipes.samplingCoins])[b] = coins[b];
                devices[0].device->writeBuffer(quantIndex, reference[pass].quantHidden.data());
                executor.forward(headSegment, graph.decoderDenoise.last);
                samples = diffusionGemmaResolveDistributedSamples(
                    reinterpret_cast<float *>(execution.pipes[graph.pipes.samplerStatsSync]),
                    reinterpret_cast<float *>(execution.pipes[graph.pipes.sampledTokensSync]),
                    batches, NN_VOCAB_MICROSHARDS, graph.net.nNodes);
            } else {
                samples = runtime.denoise(canvas, 3u, 3u, 0.8f, coins, 0u, 2u - pass, pass != 0u);
            }
            std::printf("run %u pass %u: %s %.3f ms\n", repeat + 1u, pass + 1u,
                replayHead ? "head + sampling" : "decoder",
                executor.getTotalTime(STEP_EXECUTE_OP) / 1000.0);
            const float *logits = reinterpret_cast<const float *>(execution.pipes[graph.pipes.localLogits]);
            const float *mix = reinterpret_cast<const float *>(execution.pipes[graph.pipes.selfConditioning]);
            const std::size_t logitCount = static_cast<std::size_t>(batches) * h.vocabSize;
            const std::size_t mixCount = static_cast<std::size_t>(batches) * h.hiddenSize;
            const NnUint mixWidth = graph.nodes[0].buffers[mixIndex].size.x;
            std::vector<float> localMix(graph.nodes[0].buffers[mixIndex].size.nBytes / sizeof(float));
            devices[0].device->readBuffer(mixIndex, reinterpret_cast<NnByte *>(localMix.data()));
            std::vector<float> normalizedMix(mixCount);
            for (NnUint b = 0u; b < batches; ++b)
                for (NnUint i = 0u; i < h.hiddenSize; ++i)
                    normalizedMix[b * h.hiddenSize + i] = h.embeddingScale *
                        localMix[b * mixWidth + i] / localMix[b * mixWidth + h.hiddenSize];
            for (std::size_t i = 0u; i < logitCount; ++i)
                if (!std::isfinite(logits[i])) throw std::runtime_error("non-finite logit");
            for (std::size_t i = 0u; i < mixCount; ++i)
                if (!std::isfinite(mix[i]) || !std::isfinite(normalizedMix[i]))
                    throw std::runtime_error("non-finite self-conditioning");
            for (const auto &sample : samples)
                if (!std::isfinite(sample.entropy))
                    throw std::runtime_error("non-finite entropy");
            if (!vulkan && repeat == 0u) {
                reference[pass].logits.assign(logits, logits + logitCount);
                reference[pass].mix.assign(mix, mix + mixCount);
                reference[pass].samples = samples;
                reference[pass].normalizedMix = normalizedMix;
                reference[pass].quantHidden.resize(graph.nodes[0].buffers[quantIndex].size.nBytes);
                devices[0].device->readBuffer(quantIndex, reference[pass].quantHidden.data());
            } else {
                if (vulkan && !replayHead) {
                    std::vector<NnByte> quant(reference[pass].quantHidden.size());
                    devices[0].device->readBuffer(quantIndex, quant.data());
                    std::vector<float> cpuHidden(mixCount), gpuHidden(mixCount);
                    dequantizeQ80toF32(reinterpret_cast<const NnBlockQ80 *>(reference[pass].quantHidden.data()),
                        cpuHidden.data(), mixCount, 1u, 0u);
                    dequantizeQ80toF32(reinterpret_cast<const NnBlockQ80 *>(quant.data()),
                        gpuHidden.data(), mixCount, 1u, 0u);
                    compare("head input (diagnostic)", gpuHidden.data(), cpuHidden, INFINITY, INFINITY);
                }
                compare("logits", logits, reference[pass].logits,
                    replayHead ? 0.001 : vulkan ? 60.0 : 0.0,
                    replayHead ? 0.0001 : vulkan ? 10.0 : 0.0);
                compare("pre-Q80 normalized mix", normalizedMix.data(), reference[pass].normalizedMix,
                    replayHead ? 0.0001 : vulkan ? 100.0 : 0.0,
                    replayHead ? 0.00001 : vulkan ? 2.0 : 0.0);
                // Q80 transport can amplify a small pre-transport difference
                // by crossing a block rounding boundary.
                compare("self-conditioning", mix, reference[pass].mix,
                    replayHead ? 0.1 : vulkan ? 100.0 : 0.0,
                    replayHead ? 0.001 : vulkan ? 2.0 : 0.0);
                NnUint changedSamples = 0u, changedArgmax = 0u;
                double entropyError = 0.0;
                for (NnUint b = 0u; b < batches; ++b) {
                    const auto &ref = reference[pass].samples[b];
                    changedSamples += samples[b].token != ref.token;
                    changedArgmax += samples[b].argmaxToken != ref.argmaxToken;
                    entropyError = std::max(entropyError,
                        std::fabs(static_cast<double>(samples[b].entropy) - ref.entropy));
                }
                std::printf("  entropy max %.9g; changed samples %u/%u, argmax %u/%u\n",
                    entropyError, changedSamples, batches, changedArgmax, batches);
                if (entropyError > (replayHead ? 0.0001 : vulkan ? 0.1 : 0.0) ||
                    (!vulkan && (changedSamples || changedArgmax)))
                    throw std::runtime_error("sampler comparison failed");
            }
            std::fflush(stdout);
        }
    }
}
} // namespace

int main(int argc, char **argv) {
    if (argc < 2 || argc > 6 || (argc == 6 &&
        std::strcmp(argv[5], "--vulkan") != 0 && std::strcmp(argv[5], "--vulkan-full") != 0)) {
        std::fprintf(stderr, "Usage: %s MODEL.ddlm [cpu-threads=8] [batches=256] [repeats=2] [--vulkan|--vulkan-full]\n", argv[0]);
        return 1;
    }
    try {
        initQuants();
        const NnUint threads = argc > 2 ? std::stoul(argv[2]) : 8u;
        const NnUint batches = argc > 3 ? std::stoul(argv[3]) : 256u;
        const NnUint repeats = argc > 4 ? std::stoul(argv[4]) : 2u;
        if (!threads || !batches || !repeats)
            throw std::invalid_argument("threads, batches and repeats must be positive");
#ifndef DLLAMA_VULKAN
        if (argc == 6)
            throw std::invalid_argument("rebuild with DLLAMA_VULKAN=1 for --vulkan");
#endif
        DiffusionGemmaModelFile model(argv[1]);
        if (model.header().weightType != DIFFUSION_GEMMA_Q40 || model.header().layerCount != 1u)
            throw std::invalid_argument("benchmark requires a converter-selected one-layer Q40 model");
        Snapshot reference[2];
        run(model, threads, batches, repeats, false, false, reference);
        if (argc == 6)
            run(model, threads, batches, repeats, true, std::strcmp(argv[5], "--vulkan") == 0, reference);
    } catch (const std::exception &error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
    return 0;
}
