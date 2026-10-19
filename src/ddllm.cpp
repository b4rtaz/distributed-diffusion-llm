#include "diffusion-gemma-app.hpp"

#include "nn/nn-executor.hpp"
#include "nn/nn-network.hpp"
#include "nn/nn-quants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

namespace {

bool stdoutIsTerminal() {
#if defined(_WIN32)
    return _isatty(_fileno(stdout)) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

void updateCanvasProgress(std::size_t resolved, std::size_t total, std::size_t *width) {
    char text[64];
    const int length = std::snprintf(text, sizeof(text), "(%zu/%zu...)", resolved, total);
    if (length < 0) return;
    const std::size_t textWidth = static_cast<std::size_t>(length);
    std::printf("\r%s", text);
    if (*width > textWidth)
        std::printf("%*s", static_cast<int>(*width - textWidth), "");
    std::fflush(stdout);
    *width = std::max(*width, textWidth);
}

void clearCanvasProgress(std::size_t *width) {
    if (*width == 0u) return;
    std::printf("\r%*s\r", static_cast<int>(*width), "");
    std::fflush(stdout);
    *width = 0u;
}

void printUsage(const char *program) {
    std::printf(
        "Usage: %s <inference|perplexity|chat|worker> [options]\n\n"
        "Root model and distribution options:\n"
        "  --model PATH                 DDLM model (required)\n"
        "  --tokenizer PATH             DDLT tokenizer (required)\n"
        "  --workers HOST:PORT [...]    remote workers; total nodes must be 1, 2, 4, or 8\n"
        "  --buffer-float-type TYPE     synchronization type: f32 for F32, q80 for Q40\n"
        "  --nbatches N                 encoder chunk size (default 32)\n"
        "  --max-seq-len N              runtime context capacity (default: min(4096, model maximum))\n"
        "  --nthreads N                 local CPU threads (default 1)\n"
        "  --net-turbo 0|1              enable low-latency sockets after loading (default 1)\n"
        "  --gpu-index N                Vulkan device index; requires DLLAMA_VULKAN build\n"
        "  --gpu-segments FROM:TO       inclusive Vulkan segment range; remaining work uses CPU\n\n"
        "Prompt and sequence options:\n"
        "  --prompt TEXT                input text (required for inference/perplexity)\n"
        "  --steps N                    inference sequence cap or per-turn chat generation cap\n\n"
        "Generation options (inference and chat):\n"
        "  --seed N                     deterministic sampler seed\n"
        "  --denoising-steps N          diffusion forwards per canvas\n"
        "  --t-min F / --t-max F        temperature schedule endpoints\n"
        "  --temperature F              constant-temperature override\n"
        "  --entropy-bound F            entropy acceptance bound\n"
        "  --stability-threshold N      prior equal argmax canvases required\n"
        "  --confidence-threshold F     mean-entropy stopping threshold\n"
        "  --topp F                     unsupported by the entropy-bound diffusion sampler\n\n"
        "Worker options:\n"
        "  --host ADDRESS               bind address (default 0.0.0.0)\n"
        "  --port N                     bind port (default 9990)\n"
        "  --nthreads N                 local CPU threads (default 1)\n"
        "  --net-turbo 0|1              enable low-latency sockets after loading (default 1)\n"
        "  --gpu-index N                Vulkan device index; requires DLLAMA_VULKAN build\n"
        "  --gpu-segments FROM:TO       inclusive Vulkan segment range; remaining work uses CPU\n",
        program);
}

std::vector<std::int32_t> toTokens(const std::vector<int> &tokens) {
    return std::vector<std::int32_t>(tokens.begin(), tokens.end());
}

struct ExecutionTotals {
    std::uint64_t executeMicros;
    std::uint64_t syncMicros;
};

ExecutionTotals encodeChunks(
    DiffusionGemmaAppContext *context,
    const std::vector<std::int32_t> &tokens,
    std::uint32_t *position) {
    ExecutionTotals totals{};
    std::size_t offset = 0u;
    while (offset < tokens.size()) {
        const std::size_t count = std::min<std::size_t>(context->promptBatchSize, tokens.size() - offset);
        std::vector<std::int32_t> chunk(tokens.begin() + offset, tokens.begin() + offset + count);
        context->runtime->encode(chunk, *position, *position);
        if (context->args->benchmark) {
            totals.executeMicros += context->executor->getTotalTime(STEP_EXECUTE_OP);
            totals.syncMicros += context->executor->getTotalTime(STEP_SYNC_NODES);
        }
        *position += static_cast<std::uint32_t>(count);
        offset += count;
    }
    return totals;
}

struct GenerationMetrics {
    std::uint32_t tokens;
    std::uint32_t decoderForwards;
    std::uint32_t canvases;
    bool eos;
    std::uint64_t encoderExecuteMicros;
    std::uint64_t encoderSyncMicros;
    std::uint64_t decoderExecuteMicros;
    std::uint64_t decoderSyncMicros;
};

GenerationMetrics generate(
    DiffusionGemmaAppContext *context,
    std::uint32_t *position,
    std::uint32_t tokenLimit,
    std::uint64_t firstCanvasIndex,
    const std::function<void(std::int32_t)> &emit) {
    DiffusionGemmaGenerationConfig config = resolveGenerationConfig(*context->header, *context->args);
    DiffusionGemmaGenerationPolicy policy(config, context->header->vocabSize,
        context->header->canvasLength, context->args->seed);
    GenerationMetrics metrics{};
    const bool showProgress = context->args->mode == "inference" && stdoutIsTerminal();
    std::size_t progressWidth = 0u;
    for (std::uint64_t canvasIndex = firstCanvasIndex; metrics.tokens < tokenLimit; ++canvasIndex) {
        if (*position + context->header->canvasLength > context->contextCapacity)
            break;
        policy.resetCanvas();
        context->runtime->resetSelfConditioning();
        std::vector<std::int32_t> current = policy.initializeCanvas(canvasIndex);
        std::vector<std::int32_t> argmax = current;
        bool hasSelfConditioning = false;
        for (std::uint32_t step = config.maxDenoisingSteps; step > 0u; --step) {
            const std::vector<double> coins = policy.samplingCoins(canvasIndex, step);
            const std::vector<DiffusionGemmaSampledPosition> sampled = context->runtime->denoise(
                current, *position, *position, policy.temperature(step), coins,
                static_cast<std::uint32_t>(canvasIndex), step, hasSelfConditioning);
            if (context->args->benchmark) {
                metrics.decoderExecuteMicros += context->executor->getTotalTime(STEP_EXECUTE_OP);
                metrics.decoderSyncMicros += context->executor->getTotalTime(STEP_SYNC_NODES);
            }
            std::vector<std::int32_t> sampledTokens(sampled.size());
            std::vector<float> entropies(sampled.size());
            for (std::size_t i = 0u; i < sampled.size(); ++i) {
                sampledTokens[i] = sampled[i].token;
                argmax[i] = sampled[i].argmaxToken;
                entropies[i] = sampled[i].entropy;
            }
            const DiffusionGemmaDenoisingResult result = policy.step(
                current, sampledTokens, argmax, entropies, step, canvasIndex);
            current = result.currentCanvas;
            argmax = result.argmaxCanvas;
            hasSelfConditioning = true;
            ++metrics.decoderForwards;
            if (showProgress)
                updateCanvasProgress(config.maxDenoisingSteps - step + 1u,
                    config.maxDenoisingSteps, &progressWidth);
            if (result.finished)
                break;
        }
        const DiffusionGemmaFinalizedCanvas finalized = policy.finalize(argmax);
        if (showProgress)
            clearCanvasProgress(&progressWidth);
        std::vector<std::int32_t> committed;
        for (std::size_t i = 0u; i < finalized.tokens.size() && metrics.tokens < tokenLimit; ++i) {
            const std::int32_t token = finalized.tokens[i];
            committed.push_back(token);
            emit(token);
            ++metrics.tokens;
            if (context->tokenizer->isEos(token)) {
                metrics.eos = true;
                break;
            }
        }
        if (!committed.empty()) {
            const ExecutionTotals totals = encodeChunks(context, committed, position);
            metrics.encoderExecuteMicros += totals.executeMicros;
            metrics.encoderSyncMicros += totals.syncMicros;
        }
        ++metrics.canvases;
        if (metrics.eos || committed.size() < context->header->canvasLength)
            break;
    }
    if (showProgress)
        clearCanvasProgress(&progressWidth);
    return metrics;
}

void inference(DiffusionGemmaAppContext *context) {
    if (context->args->prompt.empty()) throw std::runtime_error("--prompt is required");
    if (context->args->steps == 0u) throw std::runtime_error("--steps is required");
    const std::vector<std::int32_t> prompt = toTokens(context->tokenizer->encode(context->args->prompt, true));
    if (prompt.size() > context->args->steps)
        throw std::runtime_error("Prompt token count exceeds --steps");
    std::uint32_t position = 0u;
    const ExecutionTotals promptTotals = encodeChunks(context, prompt, &position);
    std::printf("%s\n", context->args->prompt.c_str());
    context->tokenizer->resetDecoder();
    const GenerationMetrics metrics = generate(context, &position, context->args->steps - position, 0u,
        [context](std::int32_t token) {
            const std::string piece = context->tokenizer->decodeToken(token, true);
            if (!piece.empty()) { std::fwrite(piece.data(), 1u, piece.size(), stdout); std::fflush(stdout); }
        });
    const std::string tail = context->tokenizer->flushDecoder();
    if (!tail.empty()) std::fwrite(tail.data(), 1u, tail.size(), stdout);
    NnSize sent = 0u, received = 0u;
    if (context->network != NULL) context->network->getStats(&sent, &received);
    std::printf("\n\nGeneration\n  tokens: %u\n  canvases: %u\n  decoder forwards: %u\n"
        "  tokens/decoder forward: %.3f\n"
        "  encoder execute/sync: %.1f / %.1f ms\n"
        "  decoder execute/sync: %.1f / %.1f ms\n"
        "  sent: %zu kB\n  received: %zu kB\n",
        metrics.tokens, metrics.canvases, metrics.decoderForwards,
        metrics.decoderForwards == 0u ? 0.0 : static_cast<double>(metrics.tokens) / metrics.decoderForwards,
        (promptTotals.executeMicros + metrics.encoderExecuteMicros) / 1000.0,
        (promptTotals.syncMicros + metrics.encoderSyncMicros) / 1000.0,
        metrics.decoderExecuteMicros / 1000.0, metrics.decoderSyncMicros / 1000.0,
        sent / 1024u, received / 1024u);
}

void perplexity(DiffusionGemmaAppContext *context) {
    if (context->args->prompt.empty()) throw std::runtime_error("--prompt is required");
    const std::vector<std::int32_t> tokens = toTokens(context->tokenizer->encode(context->args->prompt, true));
    if (tokens.size() < 2u) throw std::runtime_error("Perplexity needs at least two tokens");
    double totalLogProbability = 0.0;
    std::uint32_t position = 0u;
    while (position + 1u < tokens.size()) {
        const std::uint32_t count = std::min<std::uint32_t>(
            context->promptBatchSize, static_cast<std::uint32_t>(tokens.size() - 1u - position));
        const std::vector<std::int32_t> inputs(tokens.begin() + position, tokens.begin() + position + count);
        const std::vector<std::int32_t> targets(tokens.begin() + position + 1u, tokens.begin() + position + count + 1u);
        const std::vector<DiffusionGemmaScoredPosition> scores =
            context->runtime->score(inputs, targets, position, position);
        for (std::uint32_t i = 0u; i < count; ++i) {
            totalLogProbability += scores[i].logProbability;
            std::printf("%5u / %zu, token=%d, prob=%.9g\n", position + i + 1u,
                tokens.size() - 1u, targets[i], scores[i].probability);
        }
        position += count;
    }
    const double average = totalLogProbability / static_cast<double>(tokens.size() - 1u);
    std::printf("\nResults\n  perplexity: %.9g\n  avgLogProb: %.9g\n  bits/token: %.9g\n",
        std::exp(-average), average, -average / std::log(2.0));
}

bool readLine(const char *prompt, std::string *value) {
    std::printf("%s", prompt);
    std::fflush(stdout);
    return static_cast<bool>(std::getline(std::cin, *value));
}

void chat(DiffusionGemmaAppContext *context) {
    std::string system;
    if (!readLine("System prompt (optional): ", &system)) return;
    std::uint32_t position = 0u;
    std::uint64_t canvasIndex = 0u;
    bool first = true;
    while (position < context->contextCapacity) {
        std::string user;
        if (!readLine("\nUser\n> ", &user)) break;
        if (user.empty()) continue;
        std::vector<GemmaChatMessage> messages;
        if (first && !system.empty()) messages.push_back(GemmaChatMessage("system", system));
        messages.push_back(GemmaChatMessage("user", user));
        const std::vector<std::int32_t> prompt = toTokens(context->tokenizer->encode(
            context->tokenizer->renderChat(messages, true), first));
        encodeChunks(context, prompt, &position);
        first = false;
        const std::uint32_t limit = context->args->steps == 0u
            ? context->contextCapacity - position
            : std::min(context->args->steps, context->contextCapacity - position);
        context->tokenizer->resetDecoder();
        std::printf("\nAssistant\n");
        const GenerationMetrics metrics = generate(context, &position, limit, canvasIndex,
            [context](std::int32_t token) {
                const std::string piece = context->tokenizer->decodeToken(token, true);
                if (!piece.empty()) { std::fwrite(piece.data(), 1u, piece.size(), stdout); std::fflush(stdout); }
            });
        canvasIndex += metrics.canvases;
        const std::string tail = context->tokenizer->flushDecoder();
        if (!tail.empty()) std::fwrite(tail.data(), 1u, tail.size(), stdout);
        std::printf("\n");
        if (metrics.tokens == 0u) break;
    }
    std::printf("(end of context)\n");
}

} // namespace

int main(int argc, char **argv) {
    initQuants();
    initSockets();
    int result = EXIT_SUCCESS;
    try {
        DiffusionGemmaCliArgs args = DiffusionGemmaCliArgs::parse(argc, argv);
        if (args.help || args.mode.empty()) {
            printUsage(argv[0]);
            result = args.help ? EXIT_SUCCESS : EXIT_FAILURE;
        } else if (args.mode == "worker") {
            runDiffusionGemmaWorker(&args);
        } else if (args.mode == "inference") {
            args.benchmark = true;
            runDiffusionGemmaRoot(&args, inference);
        } else if (args.mode == "perplexity") {
            runDiffusionGemmaRoot(&args, perplexity);
        } else if (args.mode == "chat") {
            runDiffusionGemmaRoot(&args, chat);
        } else {
            throw std::runtime_error("Unsupported mode: " + args.mode);
        }
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Critical error: %s\n", error.what());
        result = EXIT_FAILURE;
    }
    cleanupSockets();
    return result;
}
