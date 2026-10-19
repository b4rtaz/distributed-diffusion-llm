#ifndef DIFFUSION_GEMMA_GENERATION_HPP
#define DIFFUSION_GEMMA_GENERATION_HPP

#include "diffusion-gemma.hpp"

#include <cstdint>
#include <vector>

struct DiffusionGemmaSampledPosition {
    std::int32_t token;
    std::int32_t argmaxToken;
    float entropy;
};

struct DiffusionGemmaScoredPosition {
    float probability;
    float logProbability;
    float entropy;
    std::int32_t argmaxToken;
};

struct DiffusionGemmaDenoisingResult {
    std::vector<std::int32_t> currentCanvas;
    std::vector<std::int32_t> argmaxCanvas;
    std::vector<bool> acceptedMask;
    bool finished;
};

struct DiffusionGemmaFinalizedCanvas {
    std::vector<std::int32_t> tokens;
    bool finished;
};

class DiffusionGemmaCounterRng {
private:
    std::uint64_t seed_;

public:
    explicit DiffusionGemmaCounterRng(std::uint64_t seed);
    std::uint64_t bits(
        std::uint64_t canvasIndex,
        std::uint64_t denoisingStep,
        std::uint64_t position,
        std::uint64_t stream) const;
    double uniform01(
        std::uint64_t canvasIndex,
        std::uint64_t denoisingStep,
        std::uint64_t position,
        std::uint64_t stream) const;
    std::uint32_t uniformInt(
        std::uint32_t upperBound,
        std::uint64_t canvasIndex,
        std::uint64_t denoisingStep,
        std::uint64_t position,
        std::uint64_t stream) const;
};

class DiffusionGemmaGenerationPolicy {
private:
    DiffusionGemmaGenerationConfig config_;
    std::uint32_t vocabSize_;
    std::uint32_t canvasLength_;
    DiffusionGemmaCounterRng rng_;
    std::vector<std::vector<std::int32_t> > argmaxHistory_;

    std::vector<bool> acceptedMask(const std::vector<float> &entropies) const;
    bool stable(const std::vector<std::int32_t> &argmaxCanvas);

public:
    DiffusionGemmaGenerationPolicy(
        const DiffusionGemmaGenerationConfig &config,
        std::uint32_t vocabSize,
        std::uint32_t canvasLength,
        std::uint64_t seed);

    float temperature(std::uint32_t remainingStep) const;
    std::vector<std::int32_t> initializeCanvas(std::uint64_t canvasIndex) const;
    std::vector<double> samplingCoins(std::uint64_t canvasIndex, std::uint32_t remainingStep) const;
    DiffusionGemmaDenoisingResult step(
        const std::vector<std::int32_t> &currentCanvas,
        const std::vector<std::int32_t> &sampledCanvas,
        const std::vector<std::int32_t> &argmaxCanvas,
        const std::vector<float> &entropies,
        std::uint32_t remainingStep,
        std::uint64_t canvasIndex);
    DiffusionGemmaFinalizedCanvas finalize(const std::vector<std::int32_t> &argmaxCanvas) const;
    void resetCanvas();
};

std::vector<DiffusionGemmaSampledPosition> diffusionGemmaSampleDenseLogits(
    const float *logits,
    std::uint32_t nPositions,
    std::uint32_t vocabSize,
    float temperature,
    const std::vector<double> &coins);

std::vector<DiffusionGemmaSampledPosition> diffusionGemmaResolveDistributedSamples(
    const float *stats,
    const float *sampledTokens,
    std::uint32_t nPositions,
    std::uint32_t nStatsShards,
    std::uint32_t nNodes);

std::vector<DiffusionGemmaScoredPosition> diffusionGemmaResolveDistributedScores(
    const float *stats,
    std::uint32_t nPositions,
    std::uint32_t nStatsShards,
    float temperature);

#endif
