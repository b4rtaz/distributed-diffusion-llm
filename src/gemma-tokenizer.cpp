#include "gemma-tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <queue>
#include <sstream>
#include <stdexcept>

namespace {

const std::uint32_t DDLT_VERSION = 1;
const std::uint32_t DDLT_ENDIAN = 0x01020304u;
const std::uint32_t DDLT_HEADER_SIZE = 60;

template <typename T>
T readValue(std::ifstream &file, const char *description) {
    T value;
    if (!file.read(reinterpret_cast<char *>(&value), sizeof(value)))
        throw std::runtime_error(std::string("Cannot read tokenizer ") + description);
    return value;
}

std::string readBlob(std::ifstream &file, std::uint64_t fileSize, const char *description) {
    const std::uint32_t length = readValue<std::uint32_t>(file, description);
    const std::streamoff position = file.tellg();
    if (position < 0 || static_cast<std::uint64_t>(position) + length > fileSize)
        throw std::runtime_error(std::string("Tokenizer ") + description + " exceeds file size");
    std::string value(length, '\0');
    if (length != 0 && !file.read(&value[0], length))
        throw std::runtime_error(std::string("Cannot read tokenizer ") + description);
    return value;
}

std::uint64_t pairKey(std::uint32_t left, std::uint32_t right) {
    return (static_cast<std::uint64_t>(left) << 32) | right;
}

bool isContinuation(unsigned char value) {
    return (value & 0xc0u) == 0x80u;
}

bool parseBytePiece(const std::string &piece, unsigned char *value) {
    if (piece.size() != 6 || piece.compare(0, 3, "<0x") != 0 || piece[5] != '>')
        return false;
    unsigned int byte = 0;
    std::istringstream parser(piece.substr(3, 2));
    parser >> std::hex >> byte;
    if (parser.fail() || !parser.eof() || byte >= 256)
        return false;
    *value = static_cast<unsigned char>(byte);
    return true;
}

bool isValidUtf8(const std::string &value) {
    for (std::size_t offset = 0; offset < value.size();) {
        const unsigned char first = static_cast<unsigned char>(value[offset]);
        std::size_t length = 0;
        if (first <= 0x7fu)
            length = 1;
        else if (first >= 0xc2u && first <= 0xdfu)
            length = 2;
        else if (first >= 0xe0u && first <= 0xefu)
            length = 3;
        else if (first >= 0xf0u && first <= 0xf4u)
            length = 4;
        else
            return false;
        if (offset + length > value.size())
            return false;
        for (std::size_t i = 1; i < length; ++i) {
            if (!isContinuation(static_cast<unsigned char>(value[offset + i])))
                return false;
        }
        if (length == 3) {
            const unsigned char second = static_cast<unsigned char>(value[offset + 1]);
            if ((first == 0xe0u && second < 0xa0u) || (first == 0xedu && second >= 0xa0u))
                return false;
        }
        if (length == 4) {
            const unsigned char second = static_cast<unsigned char>(value[offset + 1]);
            if ((first == 0xf0u && second < 0x90u) || (first == 0xf4u && second >= 0x90u))
                return false;
        }
        offset += length;
    }
    return true;
}

std::string replacementCharacter() {
    return std::string("\xef\xbf\xbd", 3);
}

struct BpeNode {
    std::uint32_t token;
    int previous;
    int next;
    std::uint32_t generation;
    bool alive;
};

struct BpeEvent {
    std::uint32_t rank;
    std::uint32_t output;
    int left;
    int right;
    std::uint32_t leftGeneration;
    std::uint32_t rightGeneration;
};

struct LaterBpeEvent {
    bool operator()(const BpeEvent &a, const BpeEvent &b) const {
        if (a.rank != b.rank)
            return a.rank > b.rank;
        return a.left > b.left;
    }
};

} // namespace

GemmaTokenizer::GemmaTokenizer(const char *path)
    : flags_(0), mergeCount_(0), byteTokenIds_() {
    if (path == NULL)
        throw std::runtime_error("Tokenizer path is null");
    std::ifstream file(path, std::ios::binary);
    if (!file)
        throw std::runtime_error(std::string("Failed to open tokenizer file: ") + path);

    char magic[4];
    if (!file.read(magic, sizeof(magic)) || std::memcmp(magic, "DDLT", 4) != 0)
        throw std::runtime_error("Invalid DDLT tokenizer magic");
    const std::uint32_t version = readValue<std::uint32_t>(file, "version");
    const std::uint32_t endian = readValue<std::uint32_t>(file, "endian marker");
    const std::uint64_t fileSize = readValue<std::uint64_t>(file, "file size");
    const std::uint32_t headerSize = readValue<std::uint32_t>(file, "header size");
    const std::uint32_t vocabSize = readValue<std::uint32_t>(file, "vocabulary size");
    mergeCount_ = readValue<std::uint32_t>(file, "merge count");
    const std::uint32_t specialCount = readValue<std::uint32_t>(file, "special-token count");
    const std::uint32_t eosCount = readValue<std::uint32_t>(file, "EOS-token count");
    const std::uint32_t metadataCount = readValue<std::uint32_t>(file, "metadata count");
    flags_ = readValue<std::uint32_t>(file, "flags");
    const std::uint32_t maxTokenBytes = readValue<std::uint32_t>(file, "maximum token length");
    const std::uint32_t templateNameLength = readValue<std::uint32_t>(file, "template-name length");
    const std::uint32_t templateLength = readValue<std::uint32_t>(file, "chat-template length");

    if (version != DDLT_VERSION)
        throw std::runtime_error("Unsupported DDLT tokenizer version: " + std::to_string(version));
    if (endian != DDLT_ENDIAN)
        throw std::runtime_error("DDLT tokenizer has the wrong byte order");
    if (headerSize != DDLT_HEADER_SIZE)
        throw std::runtime_error("Invalid DDLT tokenizer header size");
    file.seekg(0, std::ios::end);
    const std::streamoff actualSize = file.tellg();
    if (actualSize < 0 || static_cast<std::uint64_t>(actualSize) != fileSize)
        throw std::runtime_error("DDLT tokenizer file-size mismatch");
    file.seekg(headerSize, std::ios::beg);
    if (vocabSize == 0 || maxTokenBytes == 0)
        throw std::runtime_error("DDLT tokenizer has an empty vocabulary");
    if ((flags_ & BYTE_FALLBACK) == 0 || (flags_ & REPLACE_SPACE) == 0 || (flags_ & ADD_PREFIX_SPACE) != 0)
        throw std::runtime_error("Unsupported DDLT tokenizer behavior flags");

    eosTokenIds_.reserve(eosCount);
    for (std::uint32_t i = 0; i < eosCount; ++i) {
        const int token = readValue<std::int32_t>(file, "EOS token id");
        if (token < 0 || static_cast<std::uint32_t>(token) >= vocabSize)
            throw std::runtime_error("DDLT tokenizer EOS id is outside the vocabulary");
        eosTokenIds_.push_back(token);
    }

    for (std::uint32_t i = 0; i < metadataCount; ++i) {
        const std::string name = readBlob(file, fileSize, "metadata name");
        const int token = readValue<std::int32_t>(file, "metadata token id");
        if (name.empty() || token < 0 || static_cast<std::uint32_t>(token) >= vocabSize)
            throw std::runtime_error("Invalid DDLT tokenizer token metadata");
        if (!tokenIds_.insert(std::make_pair(name, token)).second)
            throw std::runtime_error("Duplicate DDLT tokenizer metadata key: " + name);
    }

    std::vector<std::uint32_t> specialIds;
    specialIds.reserve(specialCount);
    for (std::uint32_t i = 0; i < specialCount; ++i) {
        const std::uint32_t token = readValue<std::uint32_t>(file, "special token id");
        if (token >= vocabSize)
            throw std::runtime_error("DDLT special token id is outside the vocabulary");
        specialIds.push_back(token);
    }

    vocab_.reserve(vocabSize);
    specialMask_.assign(vocabSize, false);
    for (std::uint32_t i = 0; i < vocabSize; ++i) {
        std::string token = readBlob(file, fileSize, "vocabulary token");
        if (token.empty() || token.size() > maxTokenBytes)
            throw std::runtime_error("Invalid DDLT vocabulary token length");
        if (!vocabIds_.insert(std::make_pair(token, i)).second)
            throw std::runtime_error("Duplicate DDLT vocabulary token");
        vocab_.push_back(token);
    }

    for (std::uint32_t i = 0; i < specialIds.size(); ++i) {
        const std::uint32_t id = specialIds[i];
        if (specialMask_[id])
            throw std::runtime_error("Duplicate DDLT special token id");
        specialMask_[id] = true;
        SpecialToken special;
        special.piece = vocab_[id];
        special.id = id;
        specials_.push_back(special);
    }
    std::sort(specials_.begin(), specials_.end(), [](const SpecialToken &a, const SpecialToken &b) {
        if (a.piece.size() != b.piece.size())
            return a.piece.size() > b.piece.size();
        return a.id < b.id;
    });

    merges_.reserve(static_cast<std::size_t>(mergeCount_) * 2u);
    for (std::uint32_t rank = 0; rank < mergeCount_; ++rank) {
        const std::uint32_t left = readValue<std::uint32_t>(file, "merge left id");
        const std::uint32_t right = readValue<std::uint32_t>(file, "merge right id");
        const std::uint32_t output = readValue<std::uint32_t>(file, "merge output id");
        if (left >= vocabSize || right >= vocabSize || output >= vocabSize)
            throw std::runtime_error("DDLT merge token id is outside the vocabulary");
        MergeInfo info;
        info.rank = rank;
        info.output = output;
        if (!merges_.insert(std::make_pair(pairKey(left, right), info)).second)
            throw std::runtime_error("Duplicate DDLT merge pair");
        if (vocab_[left] + vocab_[right] != vocab_[output])
            throw std::runtime_error("DDLT merge output does not concatenate its inputs");
    }

    const std::streamoff templatePosition = file.tellg();
    if (templatePosition < 0 || static_cast<std::uint64_t>(templatePosition) + templateNameLength + templateLength != fileSize)
        throw std::runtime_error("DDLT tokenizer section lengths do not account for the complete file");
    chatTemplateName_.resize(templateNameLength);
    if (templateNameLength && !file.read(&chatTemplateName_[0], templateNameLength))
        throw std::runtime_error("Cannot read DDLT chat-template name");
    chatTemplate_.resize(templateLength);
    if (templateLength && !file.read(&chatTemplate_[0], templateLength))
        throw std::runtime_error("Cannot read DDLT chat template");
    if (chatTemplateName_ != "gemma4-canonical-text-v1" || chatTemplate_.empty())
        throw std::runtime_error("Unsupported or empty DDLT chat template");

    for (unsigned int byte = 0; byte < 256; ++byte) {
        char piece[7];
        std::snprintf(piece, sizeof(piece), "<0x%02X>", byte);
        const std::unordered_map<std::string, std::uint32_t>::const_iterator it = vocabIds_.find(piece);
        if (it == vocabIds_.end())
            throw std::runtime_error(std::string("Missing byte-fallback token: ") + piece);
        byteTokenIds_[byte] = it->second;
    }
    if (tokenIds_.find("bos_token_id") == tokenIds_.end() || tokenIds_.find("eos_token_id") == tokenIds_.end())
        throw std::runtime_error("DDLT tokenizer is missing BOS/EOS metadata");
}

std::vector<std::uint32_t> GemmaTokenizer::encodeOrdinary(const std::string &text) const {
    std::string normalized;
    normalized.reserve(text.size());
    for (std::size_t i = 0; i < text.size(); ++i) {
        if (text[i] == ' ')
            normalized.append("\xe2\x96\x81", 3);
        else
            normalized.push_back(text[i]);
    }

    std::vector<std::uint32_t> initial;
    for (std::size_t offset = 0; offset < normalized.size();) {
        const unsigned char first = static_cast<unsigned char>(normalized[offset]);
        std::size_t length = 1;
        if (first >= 0xc2u && first <= 0xdfu && offset + 1 < normalized.size() && isContinuation(static_cast<unsigned char>(normalized[offset + 1])))
            length = 2;
        else if (first >= 0xe0u && first <= 0xefu && offset + 2 < normalized.size() &&
                 isContinuation(static_cast<unsigned char>(normalized[offset + 1])) &&
                 isContinuation(static_cast<unsigned char>(normalized[offset + 2])))
            length = 3;
        else if (first >= 0xf0u && first <= 0xf4u && offset + 3 < normalized.size() &&
                 isContinuation(static_cast<unsigned char>(normalized[offset + 1])) &&
                 isContinuation(static_cast<unsigned char>(normalized[offset + 2])) &&
                 isContinuation(static_cast<unsigned char>(normalized[offset + 3])))
            length = 4;
        const std::string piece = normalized.substr(offset, length);
        const std::unordered_map<std::string, std::uint32_t>::const_iterator found = vocabIds_.find(piece);
        if (found != vocabIds_.end()) {
            initial.push_back(found->second);
        } else {
            for (std::size_t byte = 0; byte < length; ++byte)
                initial.push_back(byteTokenIds_[static_cast<unsigned char>(piece[byte])]);
        }
        offset += length;
    }
    if (initial.size() < 2)
        return initial;

    std::vector<BpeNode> nodes(initial.size());
    for (std::size_t i = 0; i < initial.size(); ++i) {
        nodes[i].token = initial[i];
        nodes[i].previous = i == 0 ? -1 : static_cast<int>(i - 1);
        nodes[i].next = i + 1 == initial.size() ? -1 : static_cast<int>(i + 1);
        nodes[i].generation = 0;
        nodes[i].alive = true;
    }
    std::priority_queue<BpeEvent, std::vector<BpeEvent>, LaterBpeEvent> queue;
    const auto enqueue = [this, &nodes, &queue](int left) {
        if (left < 0 || !nodes[left].alive || nodes[left].next < 0)
            return;
        const int right = nodes[left].next;
        const std::unordered_map<std::uint64_t, MergeInfo>::const_iterator merge =
            merges_.find(pairKey(nodes[left].token, nodes[right].token));
        if (merge == merges_.end())
            return;
        BpeEvent event;
        event.rank = merge->second.rank;
        event.output = merge->second.output;
        event.left = left;
        event.right = right;
        event.leftGeneration = nodes[left].generation;
        event.rightGeneration = nodes[right].generation;
        queue.push(event);
    };
    for (std::size_t i = 0; i + 1 < nodes.size(); ++i)
        enqueue(static_cast<int>(i));

    while (!queue.empty()) {
        const BpeEvent event = queue.top();
        queue.pop();
        BpeNode &left = nodes[event.left];
        BpeNode &right = nodes[event.right];
        if (!left.alive || !right.alive || left.next != event.right ||
            left.generation != event.leftGeneration || right.generation != event.rightGeneration)
            continue;
        const std::unordered_map<std::uint64_t, MergeInfo>::const_iterator current =
            merges_.find(pairKey(left.token, right.token));
        if (current == merges_.end() || current->second.rank != event.rank || current->second.output != event.output)
            continue;
        left.token = event.output;
        ++left.generation;
        left.next = right.next;
        right.alive = false;
        ++right.generation;
        if (right.next >= 0)
            nodes[right.next].previous = event.left;
        enqueue(left.previous);
        enqueue(event.left);
    }

    std::vector<std::uint32_t> result;
    for (int node = 0; node >= 0; node = nodes[node].next)
        result.push_back(nodes[node].token);
    return result;
}

std::vector<int> GemmaTokenizer::encode(const std::string &text, bool addSpecialTokens) const {
    std::vector<int> result;
    if (addSpecialTokens && (flags_ & ADD_BOS) != 0)
        result.push_back(tokenId("bos_token_id"));
    std::size_t ordinaryStart = 0;
    for (std::size_t offset = 0; offset < text.size();) {
        const SpecialToken *match = NULL;
        if (addSpecialTokens) {
            for (std::size_t i = 0; i < specials_.size(); ++i) {
                const SpecialToken &special = specials_[i];
                if (special.piece.size() <= text.size() - offset && text.compare(offset, special.piece.size(), special.piece) == 0) {
                    match = &special;
                    break;
                }
            }
        }
        if (match == NULL) {
            ++offset;
            continue;
        }
        const std::vector<std::uint32_t> ordinary = encodeOrdinary(text.substr(ordinaryStart, offset - ordinaryStart));
        result.insert(result.end(), ordinary.begin(), ordinary.end());
        result.push_back(static_cast<int>(match->id));
        offset += match->piece.size();
        ordinaryStart = offset;
    }
    const std::vector<std::uint32_t> ordinary = encodeOrdinary(text.substr(ordinaryStart));
    result.insert(result.end(), ordinary.begin(), ordinary.end());
    return result;
}

std::string GemmaTokenizer::drainByteFallback(bool final) {
    std::string result;
    std::size_t offset = 0;
    while (offset < bytePending_.size()) {
        const unsigned char first = static_cast<unsigned char>(bytePending_[offset]);
        std::size_t length = 0;
        bool validLead = true;
        if (first <= 0x7fu)
            length = 1;
        else if (first >= 0xc2u && first <= 0xdfu)
            length = 2;
        else if (first >= 0xe0u && first <= 0xefu)
            length = 3;
        else if (first >= 0xf0u && first <= 0xf4u)
            length = 4;
        else
            validLead = false;
        if (!validLead) {
            result += replacementCharacter();
            ++offset;
            continue;
        }
        if (offset + length > bytePending_.size()) {
            if (!final)
                break;
            result += replacementCharacter();
            ++offset;
            continue;
        }
        bool valid = true;
        for (std::size_t i = 1; i < length; ++i)
            valid = valid && isContinuation(static_cast<unsigned char>(bytePending_[offset + i]));
        if (valid && length == 3) {
            const unsigned char second = static_cast<unsigned char>(bytePending_[offset + 1]);
            valid = !(first == 0xe0u && second < 0xa0u) && !(first == 0xedu && second >= 0xa0u);
        }
        if (valid && length == 4) {
            const unsigned char second = static_cast<unsigned char>(bytePending_[offset + 1]);
            valid = !(first == 0xf0u && second < 0x90u) && !(first == 0xf4u && second >= 0x90u);
        }
        if (!valid) {
            result += replacementCharacter();
            ++offset;
            continue;
        }
        result.append(bytePending_, offset, length);
        offset += length;
    }
    bytePending_.erase(0, offset);
    return result;
}

std::string GemmaTokenizer::decodeToken(int token, bool skipSpecialTokens) {
    if (token < 0 || static_cast<std::size_t>(token) >= vocab_.size())
        throw std::runtime_error("Token id is outside the tokenizer vocabulary");
    if (specialMask_[token]) {
        std::string result = drainByteFallback(true);
        if (!skipSpecialTokens)
            result += vocab_[token];
        return result;
    }
    const std::string &piece = vocab_[token];
    unsigned char byte = 0;
    if (parseBytePiece(piece, &byte)) {
        bytePending_.push_back(static_cast<char>(byte));
        return drainByteFallback(false);
    }

    std::string result = drainByteFallback(true);
    std::size_t offset = 0;
    while (offset < piece.size()) {
        const std::size_t marker = piece.find("\xe2\x96\x81", offset);
        if (marker == std::string::npos) {
            result.append(piece, offset, std::string::npos);
            break;
        }
        result.append(piece, offset, marker - offset);
        result.push_back(' ');
        offset = marker + 3;
    }
    return result;
}

std::string GemmaTokenizer::flushDecoder() {
    return drainByteFallback(true);
}

void GemmaTokenizer::resetDecoder() {
    bytePending_.clear();
}

std::string GemmaTokenizer::decode(const std::vector<int> &tokens, bool skipSpecialTokens) {
    resetDecoder();
    std::string result;
    std::string fallbackRun;
    const auto flushFallbackRun = [&result, &fallbackRun]() {
        if (fallbackRun.empty())
            return;
        if (isValidUtf8(fallbackRun)) {
            result += fallbackRun;
        } else {
            for (std::size_t i = 0; i < fallbackRun.size(); ++i)
                result += replacementCharacter();
        }
        fallbackRun.clear();
    };
    for (std::size_t i = 0; i < tokens.size(); ++i) {
        const int token = tokens[i];
        if (token < 0 || static_cast<std::size_t>(token) >= vocab_.size())
            throw std::runtime_error("Token id is outside the tokenizer vocabulary");
        unsigned char byte = 0;
        if (!specialMask_[token] && parseBytePiece(vocab_[token], &byte)) {
            fallbackRun.push_back(static_cast<char>(byte));
            continue;
        }
        flushFallbackRun();
        if (specialMask_[token]) {
            if (!skipSpecialTokens)
                result += vocab_[token];
            continue;
        }
        const std::string &piece = vocab_[token];
        std::size_t offset = 0;
        while (offset < piece.size()) {
            const std::size_t marker = piece.find("\xe2\x96\x81", offset);
            if (marker == std::string::npos) {
                result.append(piece, offset, std::string::npos);
                break;
            }
            result.append(piece, offset, marker - offset);
            result.push_back(' ');
            offset = marker + 3;
        }
    }
    flushFallbackRun();
    return result;
}

bool GemmaTokenizer::isEos(int token) const {
    return std::find(eosTokenIds_.begin(), eosTokenIds_.end(), token) != eosTokenIds_.end();
}

bool GemmaTokenizer::isSpecial(int token) const {
    return token >= 0 && static_cast<std::size_t>(token) < specialMask_.size() && specialMask_[token];
}

int GemmaTokenizer::tokenId(const std::string &metadataName) const {
    const std::unordered_map<std::string, int>::const_iterator found = tokenIds_.find(metadataName);
    if (found == tokenIds_.end())
        throw std::runtime_error("Tokenizer metadata has no token named " + metadataName);
    return found->second;
}

const std::string &GemmaTokenizer::tokenPiece(int token) const {
    if (token < 0 || static_cast<std::size_t>(token) >= vocab_.size())
        throw std::runtime_error("Token id is outside the tokenizer vocabulary");
    return vocab_[token];
}

std::string GemmaTokenizer::trim(const std::string &text) {
    std::size_t begin = 0;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])))
        ++begin;
    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1])))
        --end;
    return text.substr(begin, end - begin);
}

std::string GemmaTokenizer::stripThinking(const std::string &text) {
    std::string result;
    std::size_t begin = 0;
    const std::string endMarker = "<channel|>";
    const std::string startMarker = "<|channel>";
    while (begin <= text.size()) {
        const std::size_t end = text.find(endMarker, begin);
        const std::string part = text.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::size_t start = part.find(startMarker);
        result += part.substr(0, start);
        if (end == std::string::npos)
            break;
        begin = end + endMarker.size();
    }
    return trim(result);
}

std::string GemmaTokenizer::renderChat(const std::vector<GemmaChatMessage> &messages, bool addGenerationPrompt) const {
    std::string result = tokenPiece(tokenId("bos_token_id"));
    std::size_t first = 0;
    if (!messages.empty() && (messages[0].role == "system" || messages[0].role == "developer")) {
        result += "<|turn>system\n";
        result += trim(messages[0].content);
        result += "<turn|>\n";
        first = 1;
    }
    std::string previousRole;
    for (std::size_t i = first; i < messages.size(); ++i) {
        const GemmaChatMessage &message = messages[i];
        if (message.role != "user" && message.role != "assistant")
            throw std::runtime_error("Text-only Gemma chat supports system/developer, user, and assistant roles");
        const bool model = message.role == "assistant";
        const bool continuation = model && previousRole == "assistant";
        if (!continuation)
            result += std::string("<|turn>") + (model ? "model" : "user") + "\n";
        result += model ? stripThinking(message.content) : trim(message.content);
        const bool continuesNext = model && i + 1 < messages.size() && messages[i + 1].role == "assistant";
        if (!continuesNext)
            result += "<turn|>\n";
        previousRole = message.role;
    }
    if (addGenerationPrompt)
        result += "<|turn>model\n";
    return result;
}
