#include "diffusion-gemma-app.hpp"

#include "nn/nn-cpu.hpp"
#include "nn/nn-network.hpp"
#if defined(DLLAMA_VULKAN)
#include "nn/nn-vulkan.hpp"
#endif

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <memory>
#include <stdexcept>

namespace {

const std::uint32_t DEFAULT_CONTEXT_CAPACITY = 4096u;

std::uint32_t parseUint(const char *name, const char *value, bool allowZero = false) {
    if (value == NULL || *value == '\0' || *value == '-')
        throw std::runtime_error(std::string("Invalid value for ") + name);
    errno = 0;
    char *end = NULL;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (errno != 0 || *end != '\0' || parsed > UINT32_MAX || (!allowZero && parsed == 0u))
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    return static_cast<std::uint32_t>(parsed);
}

std::uint64_t parseUint64(const char *name, const char *value) {
    if (value == NULL || *value == '\0' || *value == '-')
        throw std::runtime_error(std::string("Invalid value for ") + name);
    errno = 0;
    char *end = NULL;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    if (errno != 0 || *end != '\0')
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    return static_cast<std::uint64_t>(parsed);
}

float parseFloat(const char *name, const char *value) {
    errno = 0;
    char *end = NULL;
    const float parsed = std::strtof(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed))
        throw std::runtime_error(std::string("Invalid value for ") + name + ": " + value);
    return parsed;
}

void parseWorker(const char *value, std::string *host, std::uint32_t *port) {
    const char *separator = std::strrchr(value, ':');
    if (separator == NULL || separator == value || separator[1] == '\0')
        throw std::runtime_error(std::string("Invalid worker address: ") + value);
    *host = std::string(value, static_cast<std::size_t>(separator - value));
    *port = parseUint("worker port", separator + 1);
}

std::vector<NnExecutorDevice> resolveDevices(
    DiffusionGemmaCliArgs *args,
    NnNetConfig *net,
    NnNodeConfig *node,
    NnNetExecution *execution) {
    std::vector<NnExecutorDevice> devices;
    if (args->gpuIndex >= 0) {
#if defined(DLLAMA_VULKAN)
        devices.push_back(NnExecutorDevice(
            new NnVulkanDevice(args->gpuIndex, net, node, execution),
            args->gpuSegmentFrom,
            args->gpuSegmentTo));
#else
        throw std::runtime_error("This build does not support Vulkan");
#endif
    }
    if (args->gpuIndex < 0 || (args->gpuSegmentFrom >= 0 && args->gpuSegmentTo >= 0))
        devices.push_back(NnExecutorDevice(new NnCpuDevice(net, node, execution), -1, -1));
    return devices;
}

std::uint32_t workerContextCapacity(const NnNodeConfig &node) {
    for (NnUint i = 0u; i < node.nBuffers; ++i)
        if (std::strcmp(node.buffers[i].name, "rope_full") == 0)
            return node.buffers[i].size.y;
    throw std::runtime_error("Worker graph has no context-capacity buffer");
}

} // namespace

DiffusionGemmaCliArgs DiffusionGemmaCliArgs::parse(int argc, char **argv) {
    DiffusionGemmaCliArgs args{};
    args.host = "0.0.0.0";
    args.port = 9990u;
    args.nThreads = 1u;
    args.nBatches = 32u;
    args.seed = static_cast<std::uint64_t>(std::time(NULL));
    args.netTurbo = true;
    args.gpuIndex = -1;
    args.gpuSegmentFrom = -1;
    args.gpuSegmentTo = -1;
    if (argc < 2)
        return args;
    args.mode = argv[1];
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "--usage") == 0 ||
            std::strcmp(argv[i], "-h") == 0) {
            args.help = true;
            return args;
        }
    }
    for (int i = 2; i < argc;) {
        const char *name = argv[i++];
        if (std::strcmp(name, "--workers") == 0) {
            const int start = i;
            while (i < argc && argv[i][0] != '-') {
                std::string host;
                std::uint32_t port;
                parseWorker(argv[i++], &host, &port);
                args.workerHosts.push_back(host);
                args.workerPorts.push_back(port);
            }
            if (i == start)
                throw std::runtime_error("--workers requires at least one host:port");
            continue;
        }
        if (i >= argc)
            throw std::runtime_error(std::string("Missing value for ") + name);
        const char *value = argv[i++];
        if (std::strcmp(name, "--model") == 0) args.modelPath = value;
        else if (std::strcmp(name, "--tokenizer") == 0) args.tokenizerPath = value;
        else if (std::strcmp(name, "--prompt") == 0) args.prompt = value;
        else if (std::strcmp(name, "--host") == 0) args.host = value;
        else if (std::strcmp(name, "--port") == 0) args.port = parseUint(name, value);
        else if (std::strcmp(name, "--nthreads") == 0) args.nThreads = parseUint(name, value);
        else if (std::strcmp(name, "--nbatches") == 0) args.nBatches = parseUint(name, value);
        else if (std::strcmp(name, "--steps") == 0) args.steps = parseUint(name, value);
        else if (std::strcmp(name, "--max-seq-len") == 0) args.maxSeqLen = parseUint(name, value);
        else if (std::strcmp(name, "--seed") == 0) args.seed = parseUint64(name, value);
        else if (std::strcmp(name, "--denoising-steps") == 0) {
            args.denoisingSteps = parseUint(name, value); args.hasDenoisingSteps = true;
        } else if (std::strcmp(name, "--t-min") == 0) {
            args.tMin = parseFloat(name, value); args.hasTMin = true;
        } else if (std::strcmp(name, "--t-max") == 0) {
            args.tMax = parseFloat(name, value); args.hasTMax = true;
        } else if (std::strcmp(name, "--temperature") == 0) {
            args.tMin = args.tMax = parseFloat(name, value); args.hasTMin = args.hasTMax = true;
        } else if (std::strcmp(name, "--entropy-bound") == 0) {
            args.entropyBound = parseFloat(name, value); args.hasEntropyBound = true;
        } else if (std::strcmp(name, "--stability-threshold") == 0) {
            args.stabilityThreshold = parseUint(name, value, true); args.hasStabilityThreshold = true;
        } else if (std::strcmp(name, "--confidence-threshold") == 0) {
            args.confidenceThreshold = parseFloat(name, value); args.hasConfidenceThreshold = true;
        } else if (std::strcmp(name, "--net-turbo") == 0) args.netTurbo = parseUint(name, value, true) != 0u;
        else if (std::strcmp(name, "--gpu-index") == 0) args.gpuIndex = std::atoi(value);
        else if (std::strcmp(name, "--gpu-segments") == 0) {
            const char *separator = std::strchr(value, ':');
            if (separator == NULL) throw std::runtime_error("--gpu-segments expects FROM:TO");
            args.gpuSegmentFrom = std::atoi(value);
            args.gpuSegmentTo = std::atoi(separator + 1);
        } else if (std::strcmp(name, "--topp") == 0) {
            throw std::runtime_error("--topp is not supported by the DiffusionGemma entropy-bound sampler");
        } else if (std::strcmp(name, "--buffer-float-type") == 0) {
            if (std::strcmp(value, "f32") != 0 && std::strcmp(value, "q80") != 0)
                throw std::runtime_error("DiffusionGemma synchronization type must be f32 or q80");
            args.syncType = value;
        } else {
            throw std::runtime_error(std::string("Unknown option: ") + name);
        }
    }
    return args;
}

DiffusionGemmaGenerationConfig resolveGenerationConfig(
    const DiffusionGemmaHeader &header,
    const DiffusionGemmaCliArgs &args) {
    DiffusionGemmaGenerationConfig config = header.generation;
    if (args.hasDenoisingSteps) config.maxDenoisingSteps = args.denoisingSteps;
    if (args.hasTMin) config.tMin = args.tMin;
    if (args.hasTMax) config.tMax = args.tMax;
    if (args.hasEntropyBound) config.entropyBound = args.entropyBound;
    if (args.hasStabilityThreshold) config.stabilityThreshold = args.stabilityThreshold;
    if (args.hasConfidenceThreshold) config.confidenceThreshold = args.confidenceThreshold;
    return config;
}

void runDiffusionGemmaRoot(DiffusionGemmaCliArgs *args, DiffusionGemmaAppHandler handler) {
    if (args == NULL || handler == NULL || args->modelPath.empty() || args->tokenizerPath.empty())
        throw std::runtime_error("--model and --tokenizer are required");
    const NnUint nNodes = static_cast<NnUint>(args->workerHosts.size() + 1u);
    if (nNodes != 1u && nNodes != 2u && nNodes != 4u && nNodes != 8u)
        throw std::runtime_error("DiffusionGemma requires 1, 2, 4, or 8 total nodes");
    DiffusionGemmaModelFile model(args->modelPath.c_str());
    const std::string requiredSync = model.header().weightType == DIFFUSION_GEMMA_Q40 ? "q80" : "f32";
    if (!args->syncType.empty() && args->syncType != requiredSync)
        throw std::runtime_error("This model requires --buffer-float-type " + requiredSync);
    GemmaTokenizer tokenizer(args->tokenizerPath.c_str());
    if (tokenizer.vocabSize() != model.header().vocabSize)
        throw std::runtime_error("Tokenizer and model vocabulary sizes differ");
    const NnUint context = args->maxSeqLen == 0u
        ? std::min<NnUint>(DEFAULT_CONTEXT_CAPACITY, model.header().maxPositionEmbeddings)
        : args->maxSeqLen;
    if (context > model.header().maxPositionEmbeddings)
        throw std::runtime_error("--max-seq-len exceeds the model capacity");
    const NnUint maxBatch = std::max<NnUint>(args->nBatches, model.header().canvasLength);
    DiffusionGemmaGraph graph = buildDiffusionGemmaGraph(model.header(), nNodes, maxBatch, context);
    std::unique_ptr<DiffusionGemmaGraph, void (*)(DiffusionGemmaGraph *)> graphGuard(&graph, releaseDiffusionGemmaGraph);
    NnNetExecution execution(args->nThreads, &graph.net);
    std::printf("DiffusionGemma: %u layer(s), hidden %u, vocab %u, %u node(s), %s weights/%s sync\n",
        model.header().layerCount, model.header().hiddenSize, model.header().vocabSize, nNodes,
        model.header().weightType == DIFFUSION_GEMMA_Q40 ? "Q40" : "F32", requiredSync.c_str());
    printNodeRequiredMemory(&graph.net, &graph.nodes[0]);
    std::unique_ptr<NnNetwork> network;
    std::unique_ptr<NnNodeSynchronizer> synchronizer;
    if (nNodes == 1u) {
        synchronizer.reset(new NnFakeNodeSynchronizer());
    } else {
        std::vector<char *> hosts(args->workerHosts.size());
        for (std::size_t i = 0u; i < hosts.size(); ++i)
            hosts[i] = const_cast<char *>(args->workerHosts[i].c_str());
        network = NnNetwork::connect(static_cast<NnUint>(hosts.size()), hosts.data(), args->workerPorts.data());
        synchronizer.reset(new NnNetworkNodeSynchronizer(network.get(), &execution, &graph.net, &graph.nodes[0]));
        NnRootConfigWriter writer(network.get());
        writer.writeToWorkers(&graph.net, graph.nodes.data());
    }
    std::vector<NnExecutorDevice> devices = resolveDevices(args, &graph.net, &graph.nodes[0], &execution);
    NnExecutor executor(&graph.net, &graph.nodes[0], &devices, &execution, synchronizer.get(), args->benchmark);
    NnRootWeightLoader weights(&executor, network.get(), nNodes);
    loadDiffusionGemmaWeights(model, graph, &weights);
    RootDiffusionGemmaRuntime runtime(&graph, &execution, &executor, network.get());
    if (network.get() != NULL) {
        network->resetStats();
        if (args->netTurbo) network->setTurbo(true);
    }
    DiffusionGemmaAppContext contextValue{args, &model.header(), &tokenizer, &runtime, &executor,
        network.get(), args->nBatches, context};
    try {
        handler(&contextValue);
    } catch (...) {
        runtime.finish();
        throw;
    }
    runtime.finish();
}

void runDiffusionGemmaWorker(DiffusionGemmaCliArgs *args) {
    if (args == NULL) throw std::invalid_argument("Worker arguments are missing");
    for (;;) {
        std::unique_ptr<NnNetwork> network = NnNetwork::serve(args->host.c_str(), args->port);
        try {
            NnWorkerConfigReader reader(network.get());
            NnNetConfig net = reader.readNet();
            NnNodeConfig node = reader.readNode();
            std::unique_ptr<NnNetConfig, void (*)(NnNetConfig *)> netGuard(&net, releaseNetConfig);
            std::unique_ptr<NnNodeConfig, void (*)(NnNodeConfig *)> nodeGuard(&node, releaseNodeConfig);
            NnNetExecution execution(args->nThreads, &net);
            printNodeRequiredMemory(&net, &node);
            std::vector<NnExecutorDevice> devices = resolveDevices(args, &net, &node, &execution);
            NnNetworkNodeSynchronizer synchronizer(network.get(), &execution, &net, &node);
            NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
            NnWorkerWeightReader weights(&executor, network.get());
            weights.read();
            WorkerDiffusionGemmaRuntime runtime(&execution, &executor, network.get(), &net, node.nSegments,
                workerContextCapacity(node));
            if (args->netTurbo) network->setTurbo(true);
            while (!runtime.isFinished)
                runtime.tryRun(0ul);
        } catch (const NnTransferSocketException &error) {
            std::fprintf(stderr, "Network error: %s\n", error.what());
        } catch (const NnExecutorException &error) {
            std::fprintf(stderr, "Inference error: %s\n", error.what());
        }
    }
}
