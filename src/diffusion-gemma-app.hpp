#ifndef DIFFUSION_GEMMA_APP_HPP
#define DIFFUSION_GEMMA_APP_HPP

#include "diffusion-gemma-loader.hpp"
#include "diffusion-gemma-runtime.hpp"
#include "gemma-tokenizer.hpp"

#include <cstdint>
#include <string>
#include <vector>

class NnExecutor;
class NnNetwork;

struct DiffusionGemmaCliArgs {
    std::string mode;
    std::string modelPath;
    std::string tokenizerPath;
    std::string prompt;
    std::string syncType;
    std::string host;
    std::vector<std::string> workerHosts;
    std::vector<std::uint32_t> workerPorts;
    std::uint32_t port;
    std::uint32_t nThreads;
    std::uint32_t nBatches;
    std::uint32_t steps;
    std::uint32_t maxSeqLen;
    std::uint32_t denoisingSteps;
    std::uint64_t seed;
    float tMin;
    float tMax;
    float entropyBound;
    std::uint32_t stabilityThreshold;
    float confidenceThreshold;
    bool hasDenoisingSteps;
    bool hasTMin;
    bool hasTMax;
    bool hasEntropyBound;
    bool hasStabilityThreshold;
    bool hasConfidenceThreshold;
    bool netTurbo;
    bool help;
    bool benchmark;
    int gpuIndex;
    int gpuSegmentFrom;
    int gpuSegmentTo;

    static DiffusionGemmaCliArgs parse(int argc, char **argv);
};

struct DiffusionGemmaAppContext {
    DiffusionGemmaCliArgs *args;
    const DiffusionGemmaHeader *header;
    GemmaTokenizer *tokenizer;
    RootDiffusionGemmaRuntime *runtime;
    NnExecutor *executor;
    NnNetwork *network;
    std::uint32_t promptBatchSize;
    std::uint32_t contextCapacity;
};

typedef void (*DiffusionGemmaAppHandler)(DiffusionGemmaAppContext *context);

DiffusionGemmaGenerationConfig resolveGenerationConfig(
    const DiffusionGemmaHeader &header,
    const DiffusionGemmaCliArgs &args);
void runDiffusionGemmaRoot(DiffusionGemmaCliArgs *args, DiffusionGemmaAppHandler handler);
void runDiffusionGemmaWorker(DiffusionGemmaCliArgs *args);

#endif
