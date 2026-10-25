#include "diffusion-gemma-runtime.hpp"

#include <cassert>
#include <stdexcept>

static DiffusionGemmaControlPacket validPacket() {
    DiffusionGemmaControlPacket packet{};
    packet.magic = DIFFUSION_GEMMA_CONTROL_MAGIC;
    packet.version = DIFFUSION_GEMMA_CONTROL_VERSION;
    packet.command = DIFFUSION_GEMMA_CONTROL_FORWARD;
    packet.plan = DIFFUSION_GEMMA_DECODER_DENOISE;
    packet.queryLength = 4u;
    packet.absoluteStartPosition = 8u;
    packet.prefixLength = 8u;
    packet.segmentFrom = 8u;
    packet.segmentTo = 18u;
    packet.denoisingStep = 4u;
    packet.temperature = 1.0f;
    return packet;
}

static bool rejects(const DiffusionGemmaControlPacket &packet) {
    try {
        validateDiffusionGemmaControlPacket(packet, 4u, 19u, 16u);
        return false;
    } catch (const std::runtime_error &) {
        return true;
    }
}

int main() {
    DiffusionGemmaControlPacket packet = validPacket();
    validateDiffusionGemmaControlPacket(packet, 4u, 19u, 16u);
    packet.version++;
    assert(rejects(packet));
    packet = validPacket();
    packet.queryLength = 5u;
    assert(rejects(packet));
    packet = validPacket();
    packet.prefixLength = 7u;
    assert(rejects(packet));
    packet = validPacket();
    packet.segmentTo = 19u;
    assert(rejects(packet));
    packet = validPacket();
    packet.denoisingStep = 0u;
    assert(rejects(packet));
    packet = validPacket();
    packet.temperature = 0.0f;
    assert(rejects(packet));
    packet = validPacket();
    packet.command = DIFFUSION_GEMMA_CONTROL_STOP;
    packet.queryLength = 0u;
    validateDiffusionGemmaControlPacket(packet, 4u, 19u, 16u);
    return 0;
}
