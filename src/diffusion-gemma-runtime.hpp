#ifndef DIFFUSION_GEMMA_RUNTIME_HPP
#define DIFFUSION_GEMMA_RUNTIME_HPP

#include "diffusion-gemma-generation.hpp"
#include "diffusion-gemma.hpp"

#include <cstdint>
#include <vector>

class NnExecutor;
class NnNetExecution;
class NnNetwork;

#define DIFFUSION_GEMMA_CONTROL_MAGIC 0x44474350u
#define DIFFUSION_GEMMA_CONTROL_VERSION 2u

enum DiffusionGemmaControlCommand {
    DIFFUSION_GEMMA_CONTROL_FORWARD = 0,
    DIFFUSION_GEMMA_CONTROL_STOP = 1,
};

enum DiffusionGemmaControlFlags {
    DIFFUSION_GEMMA_CONTROL_SELF_CONDITIONING = 1u << 0,
};

struct DiffusionGemmaControlPacket {
    std::uint32_t magic;
    std::uint32_t version;
    std::uint32_t command;
    std::uint32_t plan;
    std::uint32_t queryLength;
    std::uint32_t absoluteStartPosition;
    std::uint32_t prefixLength;
    std::uint32_t segmentFrom;
    std::uint32_t segmentTo;
    std::uint32_t flags;
    std::uint32_t canvasIndex;
    std::uint32_t denoisingStep;
    float temperature;
};

void validateDiffusionGemmaControlPacket(
    const DiffusionGemmaControlPacket &packet,
    std::uint32_t maxBatchSize,
    std::uint32_t segmentCount,
    std::uint32_t contextCapacity);

class RootDiffusionGemmaRuntime {
private:
    const DiffusionGemmaGraph *graph_;
    NnNetExecution *execution_;
    NnExecutor *executor_;
    NnNetwork *network_;
    float *tokens_;
    float *positions_;
    float *prefixLength_;
    float *selfConditioning_;
    float *temperature_;
    float *samplingCoins_;
    float *targetTokens_;
    float *samplerStats_;
    float *sampledTokens_;

    void forward(
        DiffusionGemmaExecutionPlan plan,
        const std::vector<std::int32_t> &tokens,
        std::uint32_t absoluteStartPosition,
        std::uint32_t prefixLength,
        float temperature,
        const std::vector<double> *samplingCoins,
        const std::vector<std::int32_t> *targetTokens,
        std::uint32_t canvasIndex,
        std::uint32_t denoisingStep,
        bool hasSelfConditioning);

public:
    RootDiffusionGemmaRuntime(
        const DiffusionGemmaGraph *graph,
        NnNetExecution *execution,
        NnExecutor *executor,
        NnNetwork *network);

    void encode(
        const std::vector<std::int32_t> &tokens,
        std::uint32_t absoluteStartPosition,
        std::uint32_t prefixLength,
        bool withLogits = false);
    std::vector<DiffusionGemmaScoredPosition> score(
        const std::vector<std::int32_t> &tokens,
        const std::vector<std::int32_t> &targetTokens,
        std::uint32_t absoluteStartPosition,
        std::uint32_t prefixLength);
    std::vector<DiffusionGemmaSampledPosition> denoise(
        const std::vector<std::int32_t> &canvas,
        std::uint32_t absoluteStartPosition,
        std::uint32_t prefixLength,
        float temperature,
        const std::vector<double> &samplingCoins,
        std::uint32_t canvasIndex,
        std::uint32_t denoisingStep,
        bool hasSelfConditioning);
    void resetSelfConditioning();
    void finish();
};

class WorkerDiffusionGemmaRuntime {
private:
    NnNetExecution *execution_;
    NnExecutor *executor_;
    NnNetwork *network_;
    std::uint32_t segmentCount_;
    std::uint32_t contextCapacity_;
    DiffusionGemmaControlPacket packet_;
    float *tokens_;
    float *positions_;
    float *prefixLength_;
    float *selfConditioning_;
    float *temperature_;
    float *samplingCoins_;
    float *targetTokens_;
    NnSize selfConditioningBatchBytes_;

public:
    bool isFinished;
    WorkerDiffusionGemmaRuntime(
        NnNetExecution *execution,
        NnExecutor *executor,
        NnNetwork *network,
        NnNetConfig *netConfig,
        std::uint32_t segmentCount,
        std::uint32_t contextCapacity);
    bool tryRun(unsigned long maxAttempts = 10000ul);
};

#endif
