#include "diffusion-gemma-generation.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

static std::uint64_t mix64(std::uint64_t value) {
    value += UINT64_C(0x9e3779b97f4a7c15);
    value = (value ^ (value >> 30u)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27u)) * UINT64_C(0x94d049bb133111eb);
    return value ^ (value >> 31u);
}

static void requireCanvasSize(std::size_t actual, std::uint32_t expected, const char *name) {
    if (actual != expected)
        throw std::invalid_argument(std::string(name) + " has an invalid canvas length");
}

} // namespace

DiffusionGemmaCounterRng::DiffusionGemmaCounterRng(std::uint64_t seed) : seed_(seed) {}

std::uint64_t DiffusionGemmaCounterRng::bits(
    std::uint64_t canvasIndex,
    std::uint64_t denoisingStep,
    std::uint64_t position,
    std::uint64_t stream) const {
    std::uint64_t value = mix64(seed_ ^ UINT64_C(0x243f6a8885a308d3));
    value = mix64(value ^ mix64(canvasIndex));
    value = mix64(value ^ mix64(denoisingStep));
    value = mix64(value ^ mix64(position));
    return mix64(value ^ mix64(stream));
}

double DiffusionGemmaCounterRng::uniform01(
    std::uint64_t canvasIndex,
    std::uint64_t denoisingStep,
    std::uint64_t position,
    std::uint64_t stream) const {
    return static_cast<double>(bits(canvasIndex, denoisingStep, position, stream) >> 11u) *
        (1.0 / 9007199254740992.0);
}

std::uint32_t DiffusionGemmaCounterRng::uniformInt(
    std::uint32_t upperBound,
    std::uint64_t canvasIndex,
    std::uint64_t denoisingStep,
    std::uint64_t position,
    std::uint64_t stream) const {
    if (upperBound == 0u)
        throw std::invalid_argument("Random integer upper bound must be nonzero");
    return static_cast<std::uint32_t>(
        uniform01(canvasIndex, denoisingStep, position, stream) * upperBound);
}

DiffusionGemmaGenerationPolicy::DiffusionGemmaGenerationPolicy(
    const DiffusionGemmaGenerationConfig &config,
    std::uint32_t vocabSize,
    std::uint32_t canvasLength,
    std::uint64_t seed)
    : config_(config), vocabSize_(vocabSize), canvasLength_(canvasLength), rng_(seed) {
    if (vocabSize_ == 0u || canvasLength_ == 0u)
        throw std::invalid_argument("Vocabulary and canvas sizes must be nonzero");
    if (config_.maxDenoisingSteps == 0u)
        throw std::invalid_argument("Maximum denoising steps must be nonzero");
    if (!std::isfinite(config_.tMin) || !std::isfinite(config_.tMax) ||
        config_.tMin <= 0.0f || config_.tMax < config_.tMin)
        throw std::invalid_argument("Invalid temperature schedule");
    if (!std::isfinite(config_.entropyBound) || config_.entropyBound <= 0.0f)
        throw std::invalid_argument("Entropy bound must be finite and positive");
    if (!std::isfinite(config_.confidenceThreshold) || config_.confidenceThreshold <= 0.0f)
        throw std::invalid_argument("Confidence threshold must be finite and positive");
}

float DiffusionGemmaGenerationPolicy::temperature(std::uint32_t remainingStep) const {
    if (remainingStep == 0u || remainingStep > config_.maxDenoisingSteps)
        throw std::out_of_range("Denoising step is outside the configured schedule");
    return config_.tMin + (config_.tMax - config_.tMin) *
        (static_cast<float>(remainingStep) / static_cast<float>(config_.maxDenoisingSteps));
}

std::vector<std::int32_t> DiffusionGemmaGenerationPolicy::initializeCanvas(std::uint64_t canvasIndex) const {
    std::vector<std::int32_t> canvas(canvasLength_);
    for (std::uint32_t position = 0u; position < canvasLength_; ++position)
        canvas[position] = static_cast<std::int32_t>(
            rng_.uniformInt(vocabSize_, canvasIndex, config_.maxDenoisingSteps + 1u, position, 0u));
    return canvas;
}

std::vector<double> DiffusionGemmaGenerationPolicy::samplingCoins(
    std::uint64_t canvasIndex,
    std::uint32_t remainingStep) const {
    if (remainingStep == 0u || remainingStep > config_.maxDenoisingSteps)
        throw std::out_of_range("Denoising step is outside the configured schedule");
    std::vector<double> coins(canvasLength_);
    for (std::uint32_t position = 0u; position < canvasLength_; ++position)
        coins[position] = rng_.uniform01(canvasIndex, remainingStep, position, 1u);
    return coins;
}

std::vector<bool> DiffusionGemmaGenerationPolicy::acceptedMask(const std::vector<float> &entropies) const {
    requireCanvasSize(entropies.size(), canvasLength_, "Entropy vector");
    std::vector<std::pair<float, std::uint32_t> > sorted;
    sorted.reserve(canvasLength_);
    for (std::uint32_t position = 0u; position < canvasLength_; ++position) {
        if (!std::isfinite(entropies[position]) || entropies[position] < 0.0f)
            throw std::invalid_argument("Token entropy must be finite and non-negative");
        sorted.push_back(std::make_pair(entropies[position], position));
    }
    std::stable_sort(sorted.begin(), sorted.end(),
        [](const std::pair<float, std::uint32_t> &a, const std::pair<float, std::uint32_t> &b) {
            if (a.first != b.first)
                return a.first < b.first;
            return a.second < b.second;
        });
    std::vector<bool> accepted(canvasLength_, false);
    double cumulative = 0.0;
    for (std::uint32_t rank = 0u; rank < canvasLength_; ++rank) {
        cumulative += sorted[rank].first;
        if (cumulative - sorted[rank].first <= config_.entropyBound)
            accepted[sorted[rank].second] = true;
    }
    return accepted;
}

bool DiffusionGemmaGenerationPolicy::stable(const std::vector<std::int32_t> &argmaxCanvas) {
    if (config_.stabilityThreshold == 0u)
        return true;
    bool result = argmaxHistory_.size() == config_.stabilityThreshold;
    for (std::size_t i = 0u; result && i < argmaxHistory_.size(); ++i)
        result = argmaxHistory_[i] == argmaxCanvas;
    argmaxHistory_.push_back(argmaxCanvas);
    if (argmaxHistory_.size() > config_.stabilityThreshold)
        argmaxHistory_.erase(argmaxHistory_.begin());
    return result;
}

DiffusionGemmaDenoisingResult DiffusionGemmaGenerationPolicy::step(
    const std::vector<std::int32_t> &currentCanvas,
    const std::vector<std::int32_t> &sampledCanvas,
    const std::vector<std::int32_t> &argmaxCanvas,
    const std::vector<float> &entropies,
    std::uint32_t remainingStep,
    std::uint64_t canvasIndex) {
    requireCanvasSize(currentCanvas.size(), canvasLength_, "Current canvas");
    requireCanvasSize(sampledCanvas.size(), canvasLength_, "Sampled canvas");
    requireCanvasSize(argmaxCanvas.size(), canvasLength_, "Argmax canvas");
    (void)temperature(remainingStep);
    DiffusionGemmaDenoisingResult result;
    result.argmaxCanvas = argmaxCanvas;
    result.acceptedMask = acceptedMask(entropies);
    result.currentCanvas.resize(canvasLength_);
    double meanEntropy = 0.0;
    for (std::uint32_t position = 0u; position < canvasLength_; ++position) {
        if (sampledCanvas[position] < 0 || static_cast<std::uint32_t>(sampledCanvas[position]) >= vocabSize_ ||
            argmaxCanvas[position] < 0 || static_cast<std::uint32_t>(argmaxCanvas[position]) >= vocabSize_)
            throw std::invalid_argument("Sampled token is outside the vocabulary");
        const std::int32_t accepted = result.acceptedMask[position]
            ? sampledCanvas[position]
            : currentCanvas[position];
        const std::int32_t noise = static_cast<std::int32_t>(
            rng_.uniformInt(vocabSize_, canvasIndex, remainingStep, position, 2u));
        result.currentCanvas[position] = result.acceptedMask[position] ? accepted : noise;
        meanEntropy += entropies[position];
    }
    meanEntropy /= canvasLength_;
    result.finished = stable(argmaxCanvas) && meanEntropy < config_.confidenceThreshold;
    return result;
}

DiffusionGemmaFinalizedCanvas DiffusionGemmaGenerationPolicy::finalize(
    const std::vector<std::int32_t> &argmaxCanvas) const {
    requireCanvasSize(argmaxCanvas.size(), canvasLength_, "Argmax canvas");
    DiffusionGemmaFinalizedCanvas result{argmaxCanvas, false};
    for (std::uint32_t position = 0u; position < canvasLength_; ++position) {
        bool eos = false;
        for (std::size_t i = 0u; i < config_.eosTokenIds.size(); ++i)
            eos = eos || argmaxCanvas[position] == config_.eosTokenIds[i];
        if (!eos)
            continue;
        result.finished = true;
        for (std::uint32_t tail = position + 1u; tail < canvasLength_; ++tail)
            result.tokens[tail] = config_.padTokenId;
        break;
    }
    return result;
}

void DiffusionGemmaGenerationPolicy::resetCanvas() {
    argmaxHistory_.clear();
}

std::vector<DiffusionGemmaSampledPosition> diffusionGemmaSampleDenseLogits(
    const float *logits,
    std::uint32_t nPositions,
    std::uint32_t vocabSize,
    float temperature,
    const std::vector<double> &coins) {
    if (logits == NULL || nPositions == 0u || vocabSize == 0u)
        throw std::invalid_argument("Dense sampling dimensions must be nonzero");
    if (!std::isfinite(temperature) || temperature <= 0.0f)
        throw std::invalid_argument("Sampling temperature must be finite and positive");
    if (coins.size() != nPositions)
        throw std::invalid_argument("Dense sampling coin count mismatch");

    std::vector<DiffusionGemmaSampledPosition> result(nPositions);
    for (std::uint32_t position = 0u; position < nPositions; ++position) {
        const float *row = logits + static_cast<std::size_t>(position) * vocabSize;
        float maximum = -std::numeric_limits<float>::infinity();
        std::uint32_t argmax = 0u;
        for (std::uint32_t token = 0u; token < vocabSize; ++token) {
            if (!std::isfinite(row[token]))
                throw std::invalid_argument("Sampling logits must be finite");
            const float scaled = row[token] / temperature;
            if (scaled > maximum) {
                maximum = scaled;
                argmax = token;
            }
        }
        double denominator = 0.0;
        double weightedShiftedLogit = 0.0;
        for (std::uint32_t token = 0u; token < vocabSize; ++token) {
            const double shifted = static_cast<double>(row[token] / temperature - maximum);
            const double weight = std::exp(shifted);
            denominator += weight;
            weightedShiftedLogit += weight * shifted;
        }
        if (!(denominator > 0.0) || !std::isfinite(denominator))
            throw std::runtime_error("Sampling softmax denominator is invalid");
        if (!(coins[position] >= 0.0 && coins[position] < 1.0))
            throw std::invalid_argument("Sampling coins must be in [0, 1)");
        const double target = coins[position] * denominator;
        double cumulative = 0.0;
        std::uint32_t sampled = vocabSize - 1u;
        for (std::uint32_t token = 0u; token < vocabSize; ++token) {
            cumulative += std::exp(static_cast<double>(row[token] / temperature - maximum));
            if (target < cumulative) {
                sampled = token;
                break;
            }
        }
        result[position].token = static_cast<std::int32_t>(sampled);
        result[position].argmaxToken = static_cast<std::int32_t>(argmax);
        result[position].entropy = static_cast<float>(
            std::log(denominator) - weightedShiftedLogit / denominator);
    }
    return result;
}

std::vector<DiffusionGemmaSampledPosition> diffusionGemmaResolveDistributedSamples(
    const float *stats,
    const float *sampledTokens,
    std::uint32_t nPositions,
    std::uint32_t nStatsShards,
    std::uint32_t nNodes) {
    if (stats == NULL || sampledTokens == NULL || nPositions == 0u || nStatsShards == 0u || nNodes == 0u)
        throw std::invalid_argument("Distributed sampling dimensions must be nonzero");
    std::vector<DiffusionGemmaSampledPosition> result(nPositions);
    for (std::uint32_t position = 0u; position < nPositions; ++position) {
        const float *row = stats + static_cast<std::size_t>(position) * nStatsShards * NN_VOCAB_STATS_WIDTH;
        double denominator = 0.0;
        double moment = 0.0;
        float maximum = -std::numeric_limits<float>::infinity();
        std::int32_t argmax = -1;
        for (std::uint32_t shard = 0u; shard < nStatsShards; ++shard) {
            const float *record = row + shard * NN_VOCAB_STATS_WIDTH;
            if (!(record[0] > 0.0f) || !std::isfinite(record[0]) || !std::isfinite(record[1]) ||
                !std::isfinite(record[2]) || !std::isfinite(record[3]) || record[3] < 0.0f ||
                std::floor(record[3]) != record[3])
                throw std::runtime_error("Invalid distributed vocabulary statistics");
            denominator += record[0];
            moment += record[1];
            const std::int32_t candidate = static_cast<std::int32_t>(record[3]);
            if (record[2] > maximum || (record[2] == maximum && (argmax < 0 || candidate < argmax))) {
                maximum = record[2];
                argmax = candidate;
            }
        }
        std::int32_t sampled = -1;
        const float *tokenRow = sampledTokens + static_cast<std::size_t>(position) * nNodes;
        for (std::uint32_t node = 0u; node < nNodes; ++node) {
            const float token = tokenRow[node];
            if (!std::isfinite(token) || std::floor(token) != token)
                throw std::runtime_error("Invalid distributed sampled token");
            if (token < 0.0f)
                continue;
            if (sampled >= 0)
                throw std::runtime_error("Multiple nodes returned a distributed sampled token");
            sampled = static_cast<std::int32_t>(token);
        }
        if (sampled < 0)
            throw std::runtime_error("No node returned a distributed sampled token");
        result[position].token = sampled;
        result[position].argmaxToken = argmax;
        result[position].entropy = static_cast<float>(std::log(denominator) - moment / denominator);
    }
    return result;
}

std::vector<DiffusionGemmaScoredPosition> diffusionGemmaResolveDistributedScores(
    const float *stats,
    std::uint32_t nPositions,
    std::uint32_t nStatsShards,
    float temperature) {
    if (stats == NULL || nPositions == 0u || nStatsShards == 0u ||
        !(temperature > 0.0f) || !std::isfinite(temperature))
        throw std::invalid_argument("Distributed scoring arguments are invalid");
    std::vector<DiffusionGemmaScoredPosition> result(nPositions);
    for (std::uint32_t position = 0u; position < nPositions; ++position) {
        const float *row = stats + static_cast<std::size_t>(position) * nStatsShards * NN_VOCAB_STATS_WIDTH;
        double denominator = 0.0;
        double moment = 0.0;
        float maximum = -std::numeric_limits<float>::infinity();
        float targetLogit = -std::numeric_limits<float>::infinity();
        std::int32_t argmax = -1;
        for (std::uint32_t shard = 0u; shard < nStatsShards; ++shard) {
            const float *record = row + shard * NN_VOCAB_STATS_WIDTH;
            if (!(record[0] > 0.0f) || !std::isfinite(record[0]) || !std::isfinite(record[1]) ||
                !std::isfinite(record[2]) || !std::isfinite(record[3]))
                throw std::runtime_error("Invalid distributed vocabulary score record");
            denominator += record[0];
            moment += record[1];
            targetLogit = std::max(targetLogit, record[4]);
            const std::int32_t candidate = static_cast<std::int32_t>(record[3]);
            if (record[2] > maximum || (record[2] == maximum && (argmax < 0 || candidate < argmax))) {
                maximum = record[2];
                argmax = candidate;
            }
        }
        if (!std::isfinite(targetLogit))
            throw std::runtime_error("Distributed target token is outside the vocabulary shards");
        const double shiftedTarget = static_cast<double>(targetLogit / temperature - maximum / temperature);
        const double logProbability = shiftedTarget - std::log(denominator);
        result[position].probability = static_cast<float>(std::exp(logProbability));
        result[position].logProbability = static_cast<float>(logProbability);
        result[position].entropy = static_cast<float>(std::log(denominator) - moment / denominator);
        result[position].argmaxToken = argmax;
    }
    return result;
}
