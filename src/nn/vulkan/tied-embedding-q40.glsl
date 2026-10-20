// Common descriptor contract for the three Q40 tied-embedding dispatches.
#extension GL_EXT_shader_8bit_storage : require
#extension GL_EXT_shader_16bit_storage : require
#extension GL_EXT_shader_explicit_arithmetic_types : require

layout(constant_id = 0) const uint N_BATCHES = 32;
layout(constant_id = 1) const uint HIDDEN_SIZE = 32;

struct BatchInfo {
    uint inputOffset;
    uint inputSizeX;
    uint outputOffset;
    uint outputSizeX;
};
struct BlockQ80 { float16_t d; int8_t qs[32]; };
struct BlockQ40 { float16_t d; uint8_t qs[16]; };

layout(binding = 0) readonly buffer inputBuffer { BlockQ80 hidden[]; };
layout(binding = 1) buffer outputBuffer { float logits[]; };
layout(binding = 2) uniform batchInfosBuffer { BatchInfo infos[N_BATCHES]; };
layout(binding = 3) readonly buffer weightBuffer { BlockQ40 embedding[]; };
layout(binding = 4) uniform configBuffer {
    uint temperaturePipeIndex;
    uint partialBufferIndex;
    uint physicalSize;
    uint tileSize;
    uint tileCount;
    uint hiddenSize;
    float logitSoftcap;
};
layout(binding = 5) buffer partialBuffer { float partials[]; };
layout(binding = 6) readonly buffer temperatureBuffer { float temperature[]; };
layout(binding = 7) buffer probabilityBuffer { float probabilities[]; };
layout(push_constant) uniform runtimeConfig { uint batchSize; };
