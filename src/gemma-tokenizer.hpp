#ifndef GEMMA_TOKENIZER_HPP
#define GEMMA_TOKENIZER_HPP

#include <array>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

struct GemmaChatMessage {
    std::string role;
    std::string content;

    GemmaChatMessage(const std::string &role, const std::string &content)
        : role(role), content(content) {}
};

class GemmaTokenizer {
public:
    enum Flags {
        BYTE_FALLBACK = 1u << 0,
        REPLACE_SPACE = 1u << 1,
        ADD_PREFIX_SPACE = 1u << 2,
        ADD_BOS = 1u << 3,
    };

    explicit GemmaTokenizer(const char *path);

    std::vector<int> encode(const std::string &text, bool addSpecialTokens = true) const;
    std::string decode(const std::vector<int> &tokens, bool skipSpecialTokens = true);
    std::string decodeToken(int token, bool skipSpecialTokens = false);
    std::string flushDecoder();
    void resetDecoder();

    bool isEos(int token) const;
    bool isSpecial(int token) const;
    int tokenId(const std::string &metadataName) const;
    const std::string &tokenPiece(int token) const;

    std::string renderChat(const std::vector<GemmaChatMessage> &messages, bool addGenerationPrompt) const;

    std::uint32_t vocabSize() const { return static_cast<std::uint32_t>(vocab_.size()); }
    std::uint32_t mergeCount() const { return mergeCount_; }
    std::uint32_t flags() const { return flags_; }
    const std::vector<int> &eosTokenIds() const { return eosTokenIds_; }
    const std::string &chatTemplateName() const { return chatTemplateName_; }
    const std::string &chatTemplate() const { return chatTemplate_; }

private:
    struct MergeInfo {
        std::uint32_t rank;
        std::uint32_t output;
    };

    struct SpecialToken {
        std::string piece;
        std::uint32_t id;
    };

    std::vector<std::uint32_t> encodeOrdinary(const std::string &text) const;
    std::string drainByteFallback(bool final);
    static std::string trim(const std::string &text);
    static std::string stripThinking(const std::string &text);

    std::uint32_t flags_;
    std::uint32_t mergeCount_;
    std::vector<std::string> vocab_;
    std::unordered_map<std::string, std::uint32_t> vocabIds_;
    std::unordered_map<std::uint64_t, MergeInfo> merges_;
    std::vector<bool> specialMask_;
    std::vector<SpecialToken> specials_;
    std::vector<int> eosTokenIds_;
    std::unordered_map<std::string, int> tokenIds_;
    std::array<std::uint32_t, 256> byteTokenIds_;
    std::string chatTemplateName_;
    std::string chatTemplate_;
    std::string bytePending_;
};

#endif
