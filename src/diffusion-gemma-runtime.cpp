#include "diffusion-gemma-runtime.hpp"

#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

void validateDiffusionGemmaControlPacket(
    const DiffusionGemmaControlPacket &packet,
    std::uint32_t maxBatchSize,
    std::uint32_t segmentCount,
    std::uint32_t contextCapacity) {
    if (packet.magic != DIFFUSION_GEMMA_CONTROL_MAGIC || packet.version != DIFFUSION_GEMMA_CONTROL_VERSION)
        throw std::runtime_error("Incompatible DiffusionGemma control packet");
    if (packet.command == DIFFUSION_GEMMA_CONTROL_STOP)
        return;
    if (packet.command != DIFFUSION_GEMMA_CONTROL_FORWARD)
        throw std::runtime_error("Unknown DiffusionGemma control command");
    if (packet.plan > DIFFUSION_GEMMA_DECODER_DENOISE)
        throw std::runtime_error("Unknown DiffusionGemma execution plan");
    if (packet.queryLength == 0u || packet.queryLength > maxBatchSize)
        throw std::runtime_error("Invalid DiffusionGemma query length");
    if (packet.segmentFrom > packet.segmentTo || packet.segmentTo >= segmentCount)
        throw std::runtime_error("Invalid DiffusionGemma segment range");
    if (packet.absoluteStartPosition > contextCapacity ||
        packet.queryLength > contextCapacity - packet.absoluteStartPosition)
        throw std::runtime_error("DiffusionGemma query exceeds context capacity");
    if (packet.prefixLength != packet.absoluteStartPosition || packet.prefixLength > contextCapacity)
        throw std::runtime_error("DiffusionGemma prefix and absolute position are inconsistent");
    if (packet.plan == DIFFUSION_GEMMA_DECODER_DENOISE && packet.denoisingStep == 0u)
        throw std::runtime_error("Decoder control packet has no denoising step");
}

RootDiffusionGemmaRuntime::RootDiffusionGemmaRuntime(
    const DiffusionGemmaGraph *graph,
    NnNetExecution *execution,
    NnExecutor *executor,
    NnNetwork *network)
    : graph_(graph), execution_(execution), executor_(executor), network_(network) {
    if (graph_ == NULL || execution_ == NULL || executor_ == NULL)
        throw std::invalid_argument("Invalid root DiffusionGemma runtime");
    tokens_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.tokens]);
    positions_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.positions]);
    prefixLength_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.prefixLength]);
    selfConditioning_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.selfConditioning]);
    temperature_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.temperature]);
    samplingCoins_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.samplingCoins]);
    targetTokens_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.targetTokens]);
    samplerStats_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.samplerStatsSync]);
    sampledTokens_ = reinterpret_cast<float *>(execution_->pipes[graph_->pipes.sampledTokensSync]);
}

void RootDiffusionGemmaRuntime::forward(
    DiffusionGemmaExecutionPlan plan,
    const std::vector<std::int32_t> &tokens,
    std::uint32_t absoluteStartPosition,
    std::uint32_t prefixLength,
    float temperature,
    const std::vector<double> *samplingCoins,
    const std::vector<std::int32_t> *targetTokens,
    std::uint32_t canvasIndex,
    std::uint32_t denoisingStep,
    bool hasSelfConditioning) {
    const DiffusionGemmaSegmentRange range = diffusionGemmaPlanRange(*graph_, plan);
    DiffusionGemmaControlPacket packet{};
    packet.magic = DIFFUSION_GEMMA_CONTROL_MAGIC;
    packet.version = DIFFUSION_GEMMA_CONTROL_VERSION;
    packet.command = DIFFUSION_GEMMA_CONTROL_FORWARD;
    packet.plan = plan;
    packet.queryLength = static_cast<std::uint32_t>(tokens.size());
    packet.absoluteStartPosition = absoluteStartPosition;
    packet.prefixLength = prefixLength;
    packet.segmentFrom = range.first;
    packet.segmentTo = range.last;
    packet.flags = hasSelfConditioning ? DIFFUSION_GEMMA_CONTROL_SELF_CONDITIONING : 0u;
    packet.canvasIndex = canvasIndex;
    packet.denoisingStep = denoisingStep;
    validateDiffusionGemmaControlPacket(packet, graph_->maxBatchSize, graph_->nodes[0].nSegments,
        graph_->contextCapacity);
    if (targetTokens != NULL && targetTokens->size() != tokens.size())
        throw std::invalid_argument("DiffusionGemma target token count mismatch");

    execution_->setBatchSize(packet.queryLength);
    for (std::uint32_t i = 0u; i < packet.queryLength; ++i) {
        const std::int32_t token = tokens[i];
        if (token < 0)
            throw std::invalid_argument("DiffusionGemma token IDs must be non-negative");
        tokens_[i] = static_cast<float>(token);
        positions_[i] = static_cast<float>(absoluteStartPosition + i);
        targetTokens_[i] = static_cast<float>(targetTokens == NULL ? token : (*targetTokens)[i]);
    }
    *prefixLength_ = static_cast<float>(prefixLength);
    *temperature_ = temperature;
    if (samplingCoins != NULL) {
        if (samplingCoins->size() != tokens.size())
            throw std::invalid_argument("DiffusionGemma sampling coin count mismatch");
        for (std::uint32_t i = 0u; i < packet.queryLength; ++i) {
            const double coin = (*samplingCoins)[i];
            if (!(coin >= 0.0 && coin < 1.0))
                throw std::invalid_argument("DiffusionGemma sampling coin must be in [0, 1)");
            samplingCoins_[i] = static_cast<float>(coin);
        }
    }
    if (network_ != NULL)
        network_->writeAll(&packet, sizeof(packet));
    executor_->forward(range.first, range.last);
}

void RootDiffusionGemmaRuntime::encode(
    const std::vector<std::int32_t> &tokens,
    std::uint32_t absoluteStartPosition,
    std::uint32_t prefixLength,
    bool withLogits) {
    forward(withLogits ? DIFFUSION_GEMMA_ENCODER_WITH_LOGITS : DIFFUSION_GEMMA_ENCODER_PREFILL,
        tokens, absoluteStartPosition, prefixLength, 1.0f, NULL, NULL, 0u, 0u, false);
}

std::vector<DiffusionGemmaScoredPosition> RootDiffusionGemmaRuntime::score(
    const std::vector<std::int32_t> &tokens,
    const std::vector<std::int32_t> &targetTokens,
    std::uint32_t absoluteStartPosition,
    std::uint32_t prefixLength) {
    forward(DIFFUSION_GEMMA_ENCODER_WITH_LOGITS, tokens, absoluteStartPosition, prefixLength, 1.0f,
        NULL, &targetTokens, 0u, 0u, false);
    return diffusionGemmaResolveDistributedScores(samplerStats_, static_cast<std::uint32_t>(tokens.size()),
        NN_VOCAB_MICROSHARDS, 1.0f);
}

std::vector<DiffusionGemmaSampledPosition> RootDiffusionGemmaRuntime::denoise(
    const std::vector<std::int32_t> &canvas,
    std::uint32_t absoluteStartPosition,
    std::uint32_t prefixLength,
    float temperature,
    const std::vector<double> &samplingCoins,
    std::uint32_t canvasIndex,
    std::uint32_t denoisingStep,
    bool hasSelfConditioning) {
    forward(DIFFUSION_GEMMA_DECODER_DENOISE, canvas, absoluteStartPosition, prefixLength, temperature,
        &samplingCoins, NULL, canvasIndex, denoisingStep, hasSelfConditioning);
    return diffusionGemmaResolveDistributedSamples(samplerStats_, sampledTokens_,
        static_cast<std::uint32_t>(canvas.size()), NN_VOCAB_MICROSHARDS, graph_->net.nNodes);
}

void RootDiffusionGemmaRuntime::resetSelfConditioning() {
    const NnSize bytes = graph_->net.pipes[graph_->pipes.selfConditioning].size.nBytes;
    std::memset(selfConditioning_, 0, bytes);
}

void RootDiffusionGemmaRuntime::finish() {
    if (network_ == NULL)
        return;
    DiffusionGemmaControlPacket packet{};
    packet.magic = DIFFUSION_GEMMA_CONTROL_MAGIC;
    packet.version = DIFFUSION_GEMMA_CONTROL_VERSION;
    packet.command = DIFFUSION_GEMMA_CONTROL_STOP;
    network_->writeAll(&packet, sizeof(packet));
}

WorkerDiffusionGemmaRuntime::WorkerDiffusionGemmaRuntime(
    NnNetExecution *execution,
    NnExecutor *executor,
    NnNetwork *network,
    std::uint32_t segmentCount,
    std::uint32_t contextCapacity)
    : execution_(execution), executor_(executor), network_(network), segmentCount_(segmentCount),
      contextCapacity_(contextCapacity), packet_(), isFinished(false) {
    if (execution_ == NULL || executor_ == NULL || network_ == NULL || segmentCount_ == 0u)
        throw std::invalid_argument("Invalid worker DiffusionGemma runtime");
}

bool WorkerDiffusionGemmaRuntime::tryRun(unsigned long maxAttempts) {
    if (!network_->tryReadWithMaxAttempts(ROOT_SOCKET_INDEX, &packet_, sizeof(packet_), maxAttempts))
        return false;
    validateDiffusionGemmaControlPacket(packet_, execution_->nBatches, segmentCount_, contextCapacity_);
    if (packet_.command == DIFFUSION_GEMMA_CONTROL_STOP) {
        isFinished = true;
        return true;
    }
    execution_->setBatchSize(packet_.queryLength);
    executor_->forward(packet_.segmentFrom, packet_.segmentTo);
    return true;
}
