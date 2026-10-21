#include "diffusion-gemma-loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

static void testModel(const char *path) {
    DiffusionGemmaModelFile model(path);
    const DiffusionGemmaHeader &header = model.header();
    require(header.layerCount > 0 && header.layerTypes.size() == header.layerCount, "layer metadata mismatch");
    require(header.activation == "gelu_pytorch_tanh", "activation mismatch");
    require(header.attentionScoreScale == 1.0f, "attention score scale mismatch");
    require(header.slidingRope.type == DIFFUSION_GEMMA_DEFAULT_ROPE, "sliding RoPE mismatch");
    require(header.fullRope.type == DIFFUSION_GEMMA_PROPORTIONAL_ROPE, "full RoPE mismatch");
    require(header.generation.maxDenoisingSteps > 0, "generation config mismatch");
    std::size_t expectedTensorCount = 7;
    for (std::size_t layer = 0; layer < header.layerTypes.size(); ++layer)
        expectedTensorCount += header.layerTypes[layer] == DIFFUSION_GEMMA_SLIDING_ATTENTION ? 23 : 22;
    require(model.tensors().size() == expectedTensorCount, "model tensor count mismatch");
    const DiffusionGemmaTensor &embedding = model.tensor("token_embedding.weight");
    require(embedding.shape.size() == 2 && embedding.shape[0] == header.vocabSize && embedding.shape[1] == header.hiddenSize, "embedding shape mismatch");
    const DiffusionGemmaTensor &mixEmbedding = model.tensor("token_embedding.mix_weight");
    require(mixEmbedding.shape.size() == 2 && mixEmbedding.shape[0] == header.hiddenSize &&
        mixEmbedding.shape[1] == header.vocabSize && mixEmbedding.type == embedding.type,
        "mix embedding shape mismatch");
    std::vector<unsigned char> bytes(static_cast<std::size_t>(std::min<std::uint64_t>(embedding.length, 64)));
    model.readTensorBytes(embedding.name, 0, bytes.data(), bytes.size());
    bool rejected = false;
    try {
        model.readTensorBytes(embedding.name, embedding.length, bytes.data(), 1);
    } catch (const std::runtime_error &) {
        rejected = true;
    }
    require(rejected, "out-of-range tensor read was accepted");
    if (header.hiddenSize == 32) {
        require(header.vocabSize == 512 && header.queryHeads == 8 && header.slidingKvHeads == 8 &&
            header.slidingHeadDim == 32 && header.expertCount == 2, "synthetic header mismatch");
        require(header.layerCount == 1 && header.layerTypes[0] == DIFFUSION_GEMMA_SLIDING_ATTENTION,
            "synthetic fixture layer mismatch");
        for (std::size_t i = 0; i < bytes.size(); ++i) require(bytes[i] == 0, "synthetic tensor data mismatch");
    } else {
        require(header.hiddenSize == 2816 && header.vocabSize == 262144 && header.expertCount == 128, "checkpoint header mismatch");
        require(header.generation.eosTokenIds == std::vector<std::int32_t>({1, 106, 50}), "checkpoint EOS metadata mismatch");
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model.ddlm> [model.ddlm ...]\n", argv[0]);
        return 2;
    }
    try {
        for (int i = 1; i < argc; ++i) testModel(argv[i]);
        std::puts("DiffusionGemma loader tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "DiffusionGemma loader test failed: %s\n", error.what());
        return 1;
    }
}
