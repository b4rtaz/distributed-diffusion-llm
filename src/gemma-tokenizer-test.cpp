#include "gemma-tokenizer.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool condition, const char *message) {
    if (!condition)
        throw std::runtime_error(message);
}

static void requireTokens(const std::vector<int> &actual, const std::vector<int> &expected, const char *message) {
    if (actual != expected) {
        std::fprintf(stderr, "%s\nexpected:", message);
        for (std::size_t i = 0; i < expected.size(); ++i)
            std::fprintf(stderr, " %d", expected[i]);
        std::fprintf(stderr, "\nactual:  ");
        for (std::size_t i = 0; i < actual.size(); ++i)
            std::fprintf(stderr, " %d", actual[i]);
        std::fprintf(stderr, "\n");
        throw std::runtime_error(message);
    }
}

static void testFixture(GemmaTokenizer &tokenizer) {
    const int firstRegular = 7 + 256;
    const int idA = firstRegular;
    const int idB = firstRegular + 1;
    const int idHello = firstRegular + 10;
    const int idSpaceHello = firstRegular + 11;
    requireTokens(tokenizer.encode("hello"), std::vector<int>(1, idHello), "fixture hello encoding mismatch");
    requireTokens(tokenizer.encode(" hello"), std::vector<int>(1, idSpaceHello), "fixture leading-space encoding mismatch");
    requireTokens(tokenizer.encode("a<|turn>b"), std::vector<int>{idA, 5, idB}, "fixture special-token encoding mismatch");
    require(tokenizer.decode(std::vector<int>{idSpaceHello}) == " hello", "fixture decoding mismatch");
    require(tokenizer.decode(std::vector<int>{2, 5, 6, 1}, false) == "<bos><|turn><turn|><eos>", "fixture special decode mismatch");
    require(tokenizer.decode(std::vector<int>{2, 5, 6, 1}, true).empty(), "fixture special skipping mismatch");
    require(tokenizer.eosTokenIds() == std::vector<int>({1, 6}), "fixture EOS list mismatch");

    tokenizer.resetDecoder();
    const int byteBase = 7;
    require(tokenizer.decodeToken(byteBase + 0xf0).empty(), "streaming decoder emitted an incomplete sequence");
    require(tokenizer.decodeToken(byteBase + 0x90).empty(), "streaming decoder emitted an incomplete sequence");
    require(tokenizer.decodeToken(byteBase + 0x8d).empty(), "streaming decoder emitted an incomplete sequence");
    require(tokenizer.decodeToken(byteBase + 0x88) == "𐍈", "streaming UTF-8 decoder mismatch");
    tokenizer.resetDecoder();
    require(tokenizer.decodeToken(byteBase + 0xc3).empty(), "streaming decoder emitted an incomplete sequence");
    require(tokenizer.flushDecoder() == "�", "invalid UTF-8 recovery mismatch");

    std::vector<GemmaChatMessage> messages;
    messages.push_back(GemmaChatMessage("system", "  Be concise.  "));
    messages.push_back(GemmaChatMessage("user", "  Hello  "));
    messages.push_back(GemmaChatMessage("assistant", "<|channel>thought\nhidden<channel|> Answer "));
    const std::string expected =
        "<bos><|turn>system\nBe concise.<turn|>\n"
        "<|turn>user\nHello<turn|>\n"
        "<|turn>model\nAnswer<turn|>\n"
        "<|turn>model\n";
    require(tokenizer.renderChat(messages, true) == expected, "text chat rendering mismatch");
}

static void testRealTokenizer(GemmaTokenizer &tokenizer) {
    requireTokens(tokenizer.encode("hello"), std::vector<int>{23391}, "real ASCII encoding mismatch");
    requireTokens(tokenizer.encode("  hello  world  "), std::vector<int>{138, 23391, 138, 12392, 138}, "real whitespace encoding mismatch");
    requireTokens(tokenizer.encode("Zażółć gęślą jaźń"), std::vector<int>{236953, 40512, 24875, 237289, 549, 237202, 62081, 237198, 4828, 238992, 238041}, "real multilingual encoding mismatch");
    requireTokens(tokenizer.encode("你好，世界"), std::vector<int>{144626, 236900, 12811}, "real CJK encoding mismatch");
    requireTokens(tokenizer.encode("🙂🚀 café"), std::vector<int>{240731, 242015, 33443}, "real emoji encoding mismatch");
    requireTokens(tokenizer.encode("𐍈"), std::vector<int>{478, 382, 379, 374}, "real byte-fallback encoding mismatch");
    requireTokens(tokenizer.encode("a<|tool_call>b<tool_call|>c"), std::vector<int>{236746, 48, 236763, 49, 236755}, "real special encoding mismatch");
    requireTokens(tokenizer.encode("x<|image|>y"), std::vector<int>{236781, 258880, 236762}, "real longest-special encoding mismatch");
    const std::string text = "Zażółć gęślą jaźń 你好，世界 🙂🚀 café";
    require(tokenizer.decode(tokenizer.encode(text)) == text, "real multilingual round trip mismatch");
    require(tokenizer.decode(std::vector<int>{478, 382, 379, 374}) == "𐍈", "real byte-fallback decode mismatch");
    require(tokenizer.decode(std::vector<int>{433}) == "�", "real invalid-byte decode mismatch");
    require(tokenizer.eosTokenIds() == std::vector<int>({1, 106, 50}), "real generation EOS list mismatch");

    std::vector<GemmaChatMessage> messages;
    messages.push_back(GemmaChatMessage("system", " Be concise. "));
    messages.push_back(GemmaChatMessage("user", " Hello "));
    messages.push_back(GemmaChatMessage("assistant", " Answer "));
    messages.push_back(GemmaChatMessage("user", " Next "));
    const std::string expected =
        "<bos><|turn>system\nBe concise.<turn|>\n"
        "<|turn>user\nHello<turn|>\n"
        "<|turn>model\nAnswer<turn|>\n"
        "<|turn>user\nNext<turn|>\n"
        "<|turn>model\n";
    require(tokenizer.renderChat(messages, true) == expected, "real canonical chat rendering mismatch");
}

int main(int argc, char **argv) {
    if (argc != 2 && argc != 3) {
        std::fprintf(stderr, "Usage: %s <tokenizer.ddlt> [--probe]\n", argv[0]);
        return 2;
    }
    try {
        GemmaTokenizer tokenizer(argv[1]);
        if (argc == 3) {
            const std::string mode = argv[2];
            if (mode != "--probe" && mode != "--decode-probe")
                throw std::runtime_error("Unknown tokenizer test option");
            std::string line;
            while (std::getline(std::cin, line)) {
                if (mode == "--probe") {
                    if (line.size() % 2 != 0)
                        throw std::runtime_error("Probe input must contain complete hexadecimal bytes");
                    std::string input;
                    for (std::size_t i = 0; i < line.size(); i += 2) {
                        unsigned int byte = 0;
                        if (std::sscanf(line.substr(i, 2).c_str(), "%02x", &byte) != 1)
                            throw std::runtime_error("Probe input contains invalid hexadecimal data");
                        input.push_back(static_cast<char>(byte));
                    }
                    const std::vector<int> tokens = tokenizer.encode(input);
                    for (std::size_t i = 0; i < tokens.size(); ++i) {
                        if (i != 0)
                            std::cout << ',';
                        std::cout << tokens[i];
                    }
                } else {
                    std::vector<int> tokens;
                    std::size_t begin = 0;
                    while (begin < line.size()) {
                        const std::size_t end = line.find(',', begin);
                        tokens.push_back(std::stoi(line.substr(begin, end - begin)));
                        if (end == std::string::npos)
                            break;
                        begin = end + 1;
                    }
                    const std::string decoded = tokenizer.decode(tokens, false);
                    char byte[3];
                    for (std::size_t i = 0; i < decoded.size(); ++i) {
                        std::snprintf(byte, sizeof(byte), "%02x", static_cast<unsigned char>(decoded[i]));
                        std::cout << byte;
                    }
                }
                std::cout << '\n';
            }
            return 0;
        }
        if (tokenizer.vocabSize() == 262144)
            testRealTokenizer(tokenizer);
        else
            testFixture(tokenizer);
        std::puts("Gemma tokenizer tests passed");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "Gemma tokenizer test failed: %s\n", error.what());
        return 1;
    }
}
