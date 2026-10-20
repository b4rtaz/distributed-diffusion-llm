#ifndef DIFFUSION_GEMMA_LOADER_HPP
#define DIFFUSION_GEMMA_LOADER_HPP

#include "diffusion-gemma.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

class NnRootWeightLoader;
struct MmapFile;

class DiffusionGemmaModelFile {
public:
    explicit DiffusionGemmaModelFile(const char *path);
    ~DiffusionGemmaModelFile();

    const DiffusionGemmaHeader &header() const { return header_; }
    const std::vector<DiffusionGemmaTensor> &tensors() const { return tensors_; }
    const DiffusionGemmaTensor &tensor(const std::string &name) const;
    void readTensor(const std::string &name, void *destination, std::size_t bytes) const;
    void readTensorBytes(const std::string &name, std::uint64_t offset, void *destination, std::size_t bytes) const;

private:
    void validateManifest() const;

    std::string path_;
    std::unique_ptr<MmapFile> mappedFile_;
    DiffusionGemmaHeader header_;
    std::vector<DiffusionGemmaTensor> tensors_;
    std::unordered_map<std::string, std::size_t> tensorIndexes_;
};

void loadDiffusionGemmaWeights(
    const DiffusionGemmaModelFile &model,
    const DiffusionGemmaGraph &graph,
    NnRootWeightLoader *loader);

#endif
