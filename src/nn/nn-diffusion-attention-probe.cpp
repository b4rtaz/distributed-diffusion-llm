#include "nn-cpu-ops.cpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T>
void readValue(std::istream &input, T *value, const char *name) {
    if (!(input >> *value))
        throw std::runtime_error(std::string("Cannot read attention fixture ") + name);
}

void readVector(std::istream &input, std::vector<float> *values, const char *name) {
    for (std::size_t i = 0u; i < values->size(); ++i)
        readValue(input, &(*values)[i], name);
}

void run(const char *inputPath, const char *outputPath) {
    std::ifstream input(inputPath);
    if (!input)
        throw std::runtime_error(std::string("Cannot open attention fixture: ") + inputPath);
    std::string magic;
    readValue(input, &magic, "magic");
    if (magic != "DDAF1")
        throw std::runtime_error("Unsupported attention fixture format");

    NnUint modeValue = 0u;
    NnDiffusionAttentionOpConfig config = {};
    NnUint queryLength = 0u;
    NnUint prefixLength = 0u;
    readValue(input, &modeValue, "mode");
    readValue(input, &queryLength, "query length");
    readValue(input, &config.nQueryHeads, "query heads");
    readValue(input, &config.nKvHeads, "KV heads");
    readValue(input, &config.globalNQueryHeads, "global query heads");
    readValue(input, &config.globalNKvHeads, "global KV heads");
    readValue(input, &config.queryHeadStart, "query head start");
    readValue(input, &config.kvHeadStart, "KV head start");
    readValue(input, &config.headDim, "head dimension");
    readValue(input, &config.slidingWindow, "sliding window");
    readValue(input, &config.cacheCapacity, "cache capacity");
    readValue(input, &prefixLength, "prefix length");
    readValue(input, &config.scoreScale, "score scale");
    if (modeValue > (NnUint)DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL)
        throw std::runtime_error("Invalid attention fixture mode");
    config.mode = (NnDiffusionAttentionMode)modeValue;
    if (queryLength == 0u || config.nQueryHeads == 0u || config.nKvHeads == 0u ||
        config.globalNKvHeads == 0u || config.headDim == 0u || config.cacheCapacity == 0u)
        throw std::runtime_error("Invalid zero attention fixture dimension");
    if (config.globalNQueryHeads % config.globalNKvHeads != 0u)
        throw std::runtime_error("Invalid attention fixture GQA ratio");
    if (config.slidingWindow == 0u && prefixLength > config.cacheCapacity)
        throw std::runtime_error("Full-attention fixture prefix exceeds cache capacity");

    const NnUint queryDim = config.nQueryHeads * config.headDim;
    const NnUint kvDim = config.nKvHeads * config.headDim;
    std::vector<float> positions(queryLength);
    std::vector<float> query(queryLength * queryDim);
    std::vector<float> currentKey(queryLength * kvDim);
    std::vector<float> currentValue(queryLength * kvDim);
    std::vector<float> keyCache(config.cacheCapacity * kvDim);
    std::vector<float> valueCache(config.cacheCapacity * kvDim);
    std::vector<float> output(queryLength * queryDim);
    readVector(input, &positions, "positions");
    readVector(input, &query, "query");
    readVector(input, &currentKey, "current key");
    readVector(input, &currentValue, "current value");
    readVector(input, &keyCache, "key cache");
    readVector(input, &valueCache, "value cache");
    std::string trailing;
    if (input >> trailing)
        throw std::runtime_error("Trailing attention fixture data");

    diffusionAttention_F32(
        output.data(), query.data(), currentKey.data(), currentValue.data(),
        keyCache.data(), valueCache.data(), positions.data(), prefixLength,
        queryLength, &config, 1u, 0u);

    std::ofstream outputFile(outputPath, std::ios::out | std::ios::trunc);
    if (!outputFile)
        throw std::runtime_error(std::string("Cannot create attention result: ") + outputPath);
    outputFile << std::setprecision(9);
    for (std::size_t i = 0u; i < output.size(); ++i)
        outputFile << output[i] << (i + 1u == output.size() ? '\n' : ' ');
    if (!outputFile)
        throw std::runtime_error("Cannot write attention result");
}

} // namespace

int main(int argc, char **argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " INPUT.ddaf OUTPUT.txt\n";
        return 1;
    }
    try {
        run(argv[1], argv[2]);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 2;
    }
}
