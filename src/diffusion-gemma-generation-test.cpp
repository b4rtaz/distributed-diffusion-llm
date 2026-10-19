#include "diffusion-gemma-generation.hpp"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

static DiffusionGemmaGenerationConfig config() {
    DiffusionGemmaGenerationConfig value{};
    value.maxDenoisingSteps = 4u;
    value.maxNewTokens = 4u;
    value.tMin = 0.4f;
    value.tMax = 0.8f;
    value.entropyBound = 0.1f;
    value.stabilityThreshold = 1u;
    value.confidenceThreshold = 0.2f;
    value.padTokenId = 0;
    value.eosTokenIds.push_back(5);
    value.eosTokenIds.push_back(7);
    return value;
}

static void testTemperatureAndRng() {
    DiffusionGemmaGenerationPolicy policy(config(), 16u, 4u, 123u);
    assert(std::fabs(policy.temperature(4u) - 0.8f) < 1e-7f);
    assert(std::fabs(policy.temperature(2u) - 0.6f) < 1e-7f);
    assert(std::fabs(policy.temperature(1u) - 0.5f) < 1e-7f);
    const std::vector<std::int32_t> a = policy.initializeCanvas(0u);
    const std::vector<std::int32_t> b = policy.initializeCanvas(0u);
    const std::vector<std::int32_t> c = policy.initializeCanvas(1u);
    assert(a == b);
    assert(a != c);
    for (std::size_t i = 0u; i < a.size(); ++i)
        assert(a[i] >= 0 && a[i] < 16);
}

static void testDenseSampling() {
    const float logits[] = {
        0.0f, 1.0f, 2.0f, 3.0f,
        4.0f, 4.0f, 0.0f, 0.0f,
    };
    const std::vector<DiffusionGemmaSampledPosition> sampled =
        diffusionGemmaSampleDenseLogits(logits, 2u, 4u, 1.0f, std::vector<double>{0.0, 0.75});
    assert(sampled[0].token == 0);
    assert(sampled[0].argmaxToken == 3);
    assert(sampled[1].token == 1);
    assert(sampled[1].argmaxToken == 0); // deterministic lowest-token tie break
    assert(sampled[0].entropy > 0.0f && sampled[0].entropy < std::log(4.0f));
}

static void testDistributedSamplingResolution() {
    const float stats[] = {
        1.0f, -1.0f, 3.0f, 5.0f, -INFINITY,
        2.0f, -0.5f, 4.0f, 7.0f, -INFINITY,
        3.0f, -0.25f, 4.0f, 6.0f, 2.0f,
        4.0f, -0.125f, 1.0f, 12.0f, -INFINITY,
    };
    const float tokens[] = {-1.0f, 9.0f};
    const std::vector<DiffusionGemmaSampledPosition> result =
        diffusionGemmaResolveDistributedSamples(stats, tokens, 1u, 4u, 2u);
    assert(result[0].token == 9);
    assert(result[0].argmaxToken == 6); // equal max resolves to the lowest global token
    assert(std::fabs(result[0].entropy - (std::log(10.0) + 1.875 / 10.0)) < 1e-6);
    const std::vector<DiffusionGemmaScoredPosition> scores =
        diffusionGemmaResolveDistributedScores(stats, 1u, 4u, 1.0f);
    assert(std::fabs(scores[0].probability - std::exp(-2.0) / 10.0) < 1e-7);
    assert(scores[0].argmaxToken == 6);
}

static void testAcceptanceStoppingAndRenoise() {
    DiffusionGemmaGenerationConfig cfg = config();
    DiffusionGemmaGenerationPolicy policy(cfg, 16u, 4u, 9u);
    const std::vector<std::int32_t> current{1, 2, 3, 4};
    const std::vector<std::int32_t> sampled{8, 9, 10, 11};
    const std::vector<std::int32_t> argmax{8, 9, 10, 11};
    const std::vector<float> entropy{0.04f, 0.01f, 0.08f, 0.2f};
    DiffusionGemmaDenoisingResult first = policy.step(current, sampled, argmax, entropy, 4u, 0u);
    assert((first.acceptedMask == std::vector<bool>{true, true, true, false}));
    assert(first.currentCanvas[0] == 8 && first.currentCanvas[1] == 9 && first.currentCanvas[2] == 10);
    assert(first.currentCanvas[3] >= 0 && first.currentCanvas[3] < 16);
    assert(!first.finished); // no prior argmax canvas yet
    DiffusionGemmaDenoisingResult second = policy.step(first.currentCanvas, sampled, argmax, entropy, 3u, 0u);
    assert(second.finished); // same as the preceding one and mean entropy is below the threshold

    policy.resetCanvas();
    DiffusionGemmaDenoisingResult reset = policy.step(current, sampled, argmax, entropy, 2u, 1u);
    assert(!reset.finished);
}

static void testFinalize() {
    DiffusionGemmaGenerationPolicy policy(config(), 16u, 4u, 1u);
    DiffusionGemmaFinalizedCanvas noEos = policy.finalize(std::vector<std::int32_t>{1, 2, 3, 4});
    assert(!noEos.finished);
    DiffusionGemmaFinalizedCanvas eos = policy.finalize(std::vector<std::int32_t>{1, 7, 3, 5});
    assert(eos.finished);
    assert((eos.tokens == std::vector<std::int32_t>{1, 7, 0, 0}));
}

static void testValidation() {
    bool threw = false;
    try {
        DiffusionGemmaGenerationConfig invalid = config();
        invalid.tMin = 0.0f;
        DiffusionGemmaGenerationPolicy policy(invalid, 16u, 4u, 0u);
        (void)policy;
    } catch (const std::invalid_argument &) {
        threw = true;
    }
    assert(threw);
}

int main() {
    testTemperatureAndRng();
    testDenseSampling();
    testDistributedSamplingResolution();
    testAcceptanceStoppingAndRenoise();
    testFinalize();
    testValidation();
    return 0;
}
