#include "nn-config-builder.hpp"
#include "nn-cpu.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <vector>

static NnExecutor *buildExecutor(
    NnNetConfig *netConfig,
    NnNodeConfig *nodeConfig,
    NnNetExecution **execution,
    NnFakeNodeSynchronizer **synchronizer,
    std::vector<NnExecutorDevice> *devices
) {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint xPipeIndex = netBuilder.addPipe("x", size2D(F_32, 1u, 4u));

    NnNodeConfigBuilder nodeBuilder(0u);

    NnSegmentConfigBuilder tanhSegment;
    tanhSegment.addOp(
        OP_TANH, "tanh", 0u,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size0(),
        NnTanhOpCodeConfig{});
    nodeBuilder.addSegment(tanhSegment.build());

    NnSegmentConfigBuilder siluSegment;
    siluSegment.addOp(
        OP_SILU, "silu", 0u,
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        pointerBatchConfig(SRC_PIPE, xPipeIndex),
        size0(),
        NnSiluOpCodeConfig{});
    nodeBuilder.addSegment(siluSegment.build());

    *netConfig = netBuilder.build();
    *nodeConfig = nodeBuilder.build();
    *execution = new NnNetExecution(1u, netConfig);
    (*execution)->setBatchSize(1u);
    *synchronizer = new NnFakeNodeSynchronizer();
    devices->push_back(NnExecutorDevice(new NnCpuDevice(netConfig, nodeConfig, *execution), -1, -1));
    return new NnExecutor(netConfig, nodeConfig, devices, *execution, *synchronizer, false);
}

static void resetInput(NnNetExecution *execution, const float value) {
    float *x = (float *)execution->pipes[0];
    for (NnUint i = 0u; i < 4u; i++)
        x[i] = value;
}

static void testSharedWeights() {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint pipe = netBuilder.addPipe("x", size2D(F_32, 1u, 4u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint dummy = nodeBuilder.addBuffer("dummy", size1D(F_32, 1u));
    const NnUint shared = nodeBuilder.addWeight("shared", size2D(F_32, 4u, 4u));
    for (NnUint segmentIndex = 0u; segmentIndex < 2u; segmentIndex++) {
        NnSegmentConfigBuilder segment;
        segment.addOp(
            OP_MATMUL, "shared_matmul", segmentIndex,
            pointerBatchConfig(SRC_PIPE, pipe),
            pointerBatchConfig(SRC_PIPE, pipe),
            size2D(F_32, 4u, 4u),
            shared,
            NnMatmulOpConfig{0u, 0u, dummy});
        nodeBuilder.addSegment(segment.build());
    }
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    NnCpuDevice device(&net, &node, &execution);
    NnCpuDeviceSegment *a = (NnCpuDeviceSegment *)device.createSegment(0u);
    NnCpuDeviceSegment *b = (NnCpuDeviceSegment *)device.createSegment(1u);
    assert(a->opContexts[0].weight == b->opContexts[0].weight);
    float value = 42.0f;
    a->loadWeight(0u, 0u, sizeof(value), (NnByte *)&value);
    assert(((float *)b->opContexts[0].weight)[0] == value);
    delete a;
    delete b;
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testWeightMulAndImmediateScale() {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint pipe = netBuilder.addPipe("x", size2D(F_32, 1u, 4u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint multiplier = nodeBuilder.addWeight("multiplier", size1D(F_32, 4u));
    NnSegmentConfigBuilder segment;
    segment.addOp(
        OP_MUL, "weighted_mul", 0u,
        pointerBatchConfig(SRC_PIPE, pipe), pointerBatchConfig(SRC_PIPE, pipe),
        size1D(F_32, 4u), multiplier, NnMulOpCodeConfig{0u, 1u});
    segment.addOp(
        OP_SCALE, "immediate_scale", 0u,
        pointerBatchConfig(SRC_PIPE, pipe), pointerBatchConfig(SRC_PIPE, pipe),
        size0(), NnScaleOpCodeConfig{0u, 0.5f, 1u});
    nodeBuilder.addSegment(segment.build());
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    execution.setBatchSize(1u);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&net, &node, &execution), -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
    float weights[] = {1.0f, 2.0f, 3.0f, 4.0f};
    executor.loadWeight("weighted_mul", 0u, 0u, sizeof(weights), (NnByte *)weights);
    float *x = (float *)execution.pipes[pipe];
    for (NnUint i = 0u; i < 4u; ++i) x[i] = 2.0f;
    executor.forward();
    for (NnUint i = 0u; i < 4u; ++i) assert(std::fabs(x[i] - weights[i]) < 0.000001f);
    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testWeightedExpertScale() {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint input = netBuilder.addPipe("probabilities", size2D(F_32, 1u, 4u));
    const NnUint output = netBuilder.addPipe("weights", size3D(F_32, 2u, 1u, 1u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint indexes = nodeBuilder.addBuffer("indexes", size2D(F_32, 1u, 2u));
    const NnUint routeLayout = nodeBuilder.addBuffer("route_layout", size1D(F_32, 9u));
    const NnUint scales = nodeBuilder.addWeight("expert_scales", size1D(F_32, 4u));
    NnSegmentConfigBuilder segment;
    segment.addOp(
        OP_MOE_ROUTE, "weighted_gate", 0u,
        pointerBatchConfig(SRC_PIPE, input), pointerBatchConfig(SRC_PIPE, output),
        size1D(F_32, 4u), scales,
        NnMoeRouteOpConfig{2u, 1u, indexes, 0u, 1u, 1u, routeLayout});
    nodeBuilder.addSegment(segment.build());
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    execution.setBatchSize(1u);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&net, &node, &execution), -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
    float expertScales[] = {1.0f, 2.0f, 3.0f, 1.0f};
    executor.loadWeight("weighted_gate", 0u, 0u, sizeof(expertScales), (NnByte *)expertScales);
    float probabilities[] = {0.1f, 0.4f, 0.4f, 0.1f};
    std::copy(probabilities, probabilities + 4, (float *)execution.pipes[input]);
    executor.forward();
    const float *weights = (const float *)execution.pipes[output];
    assert(std::fabs(weights[0] - 1.0f) < 0.000001f);
    assert(std::fabs(weights[1] - 1.5f) < 0.000001f);
    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testPackedGroupedReduction() {
    NnNetConfigBuilder netBuilder(2u, 1u);
    const NnUint packed = netBuilder.addPipe("packed", size2D(F_32, 1u, 16u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint dense = nodeBuilder.addBuffer("dense", size2D(F_32, 1u, 4u));
    const NnUint moe = nodeBuilder.addBuffer("moe", size2D(F_32, 1u, 4u));
    const NnUint branches = nodeBuilder.addBuffer("branches", size3D(F_32, 2u, 1u, 4u));
    NnSegmentConfigBuilder segment;
    segment.addOp(OP_CAST, "pack_dense", 0u,
        pointerBatchConfig(SRC_BUFFER, dense), pointerBatchedGroupSliceConfig(SRC_PIPE, packed, 0u, 2u),
        size0(), NnCastOpCodeConfig{});
    segment.addOp(OP_CAST, "pack_moe", 0u,
        pointerBatchConfig(SRC_BUFFER, moe), pointerBatchedGroupSliceConfig(SRC_PIPE, packed, 1u, 2u),
        size0(), NnCastOpCodeConfig{});
    segment.addOp(OP_MERGE_SUM, "grouped_sum", 0u,
        pointerBatchConfig(SRC_PIPE, packed), pointerBatchConfig(SRC_BUFFER, branches),
        size0(), NnMergeSumOpCodeConfig{2u});
    segment.addOp(OP_SCALE, "scale_dense", 0u,
        pointerBatchZConfig(SRC_BUFFER, branches, 0u), pointerBatchZConfig(SRC_BUFFER, branches, 0u),
        size0(), NnScaleOpCodeConfig{0u, 10.0f, 1u});
    segment.addOp(OP_SCALE, "scale_moe", 0u,
        pointerBatchZConfig(SRC_BUFFER, branches, 1u), pointerBatchZConfig(SRC_BUFFER, branches, 1u),
        size0(), NnScaleOpCodeConfig{0u, 100.0f, 1u});
    nodeBuilder.addSegment(segment.build());
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    execution.setBatchSize(1u);
    NnCpuDevice *device = new NnCpuDevice(&net, &node, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
    std::fill((float *)device->buffers[dense], (float *)device->buffers[dense] + 4, 1.0f);
    std::fill((float *)device->buffers[moe], (float *)device->buffers[moe] + 4, 2.0f);
    float *pipe = (float *)execution.pipes[packed];
    std::fill(pipe + 8, pipe + 12, 3.0f);
    std::fill(pipe + 12, pipe + 16, 4.0f);
    executor.forward();
    const float *result = (const float *)device->buffers[branches];
    for (NnUint i = 0u; i < 4u; ++i) assert(std::fabs(result[i] - 40.0f) < 0.000001f);
    for (NnUint i = 4u; i < 8u; ++i) assert(std::fabs(result[i] - 600.0f) < 0.000001f);
    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testPackedGroupedQuantizedReduction() {
    constexpr NnUint width = 32u;
    constexpr NnUint groups = 2u;
    constexpr NnUint contributions = 2u;
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint packed = netBuilder.addPipe("packed_q80", size2D(F_Q80, 1u, width * groups * contributions));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint branches = nodeBuilder.addBuffer("branches_q80", size3D(F_32, groups, 1u, width));
    NnSegmentConfigBuilder segment;
    segment.addOp(OP_MERGE_SUM, "grouped_sum_q80", 0u,
        pointerBatchConfig(SRC_PIPE, packed), pointerBatchConfig(SRC_BUFFER, branches),
        size0(), NnMergeSumOpCodeConfig{groups});
    nodeBuilder.addSegment(segment.build());
    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    execution.setBatchSize(1u);
    NnCpuDevice *device = new NnCpuDevice(&net, &node, &execution);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(device, -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);
    std::vector<float> input(width * groups * contributions);
    for (NnUint i = 0u; i < width; ++i) {
        input[i] = 1.0f;
        input[width + i] = 2.0f;
        input[2u * width + i] = 3.0f;
        input[3u * width + i] = 4.0f;
    }
    quantizeF32toQ80(input.data(), (NnBlockQ80 *)execution.pipes[packed], input.size(), 1u, 0u);
    executor.forward();
    const float *result = (const float *)device->buffers[branches];
    for (NnUint i = 0u; i < width; ++i) assert(std::fabs(result[i] - 4.0f) < 0.05f);
    for (NnUint i = width; i < 2u * width; ++i) assert(std::fabs(result[i] - 6.0f) < 0.05f);
    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testPackedExpertMatmulAndMerge() {
    const NnUint maxBatches = 8u;
    const NnUint batchSize = 7u;
    const NnUint nExperts = 2u;
    const NnUint nActiveExperts = 2u;
    const NnUint dim = 4u;

    NnNetConfigBuilder netBuilder(1u, maxBatches);
    const NnUint scoresPipe = netBuilder.addPipe("scores", size2D(F_32, maxBatches, nExperts));
    const NnUint inputPipe = netBuilder.addPipe("input", size2D(F_32, maxBatches, dim));
    const NnUint outputPipe = netBuilder.addPipe("output", size2D(F_32, maxBatches, dim));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint indexes = nodeBuilder.addBuffer(
        "expert_indexes", size2D(F_32, maxBatches, nActiveExperts));
    const NnUint weights = nodeBuilder.addBuffer(
        "expert_weights", size3D(F_32, nActiveExperts, maxBatches, 1u));
    const NnUint routeLayout = nodeBuilder.addBuffer("expert_route_layout",
        size1D(F_32, 2u * maxBatches * nActiveExperts + nExperts + 1u));
    const NnUint packedInput = nodeBuilder.addBuffer(
        "packed_input", size3D(F_32, nActiveExperts, maxBatches, dim));
    const NnUint packedOutput = nodeBuilder.addBuffer(
        "packed_output", size3D(F_32, nActiveExperts, maxBatches, dim));
    const NnUint expertWeight = nodeBuilder.addWeight(
        "expert_weight", size3D(F_32, nExperts, dim, dim));

    NnSegmentConfigBuilder segment;
    segment.addOp(OP_MOE_ROUTE, "moe_route", 0u,
        pointerBatchConfig(SRC_PIPE, scoresPipe), pointerBatchConfig(SRC_BUFFER, weights),
        size0(), NnMoeRouteOpConfig{
            nActiveExperts, 0u, indexes, indexes, 0u, 0u, routeLayout});
    segment.addOp(OP_MOE_PACK, "moe_pack", 0u,
        pointerBatchConfig(SRC_PIPE, inputPipe), pointerBatchConfig(SRC_BUFFER, packedInput),
        size0(), NnMoePackOpConfig{routeLayout});
    segment.addOp(OP_MATMUL, "expert_matmul", 0u,
        pointerBatchConfig(SRC_BUFFER, packedInput), pointerBatchConfig(SRC_BUFFER, packedOutput),
        size3D(F_32, nExperts, dim, dim), expertWeight,
        NnMatmulOpConfig{nExperts, nActiveExperts, indexes, routeLayout});
    segment.addOp(OP_MOE_MERGE, "moe_merge", 0u,
        pointerBatchConfig(SRC_BUFFER, packedOutput), pointerBatchConfig(SRC_PIPE, outputPipe),
        size0(), NnMoeMergeOpConfig{nActiveExperts, weights, routeLayout});
    nodeBuilder.addSegment(segment.build());

    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(4u, &net);
    execution.setBatchSize(batchSize);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&net, &node, &execution), -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);

    float expertWeights[nExperts * dim * dim] = {};
    for (NnUint expert = 0u; expert < nExperts; ++expert)
        for (NnUint d = 0u; d < dim; ++d)
            expertWeights[expert * dim * dim + d * dim + d] = (float)(expert + 1u);
    executor.loadWeight("expert_matmul", 0u, 0u,
        sizeof(expertWeights), (NnByte *)expertWeights);

    float *scores = (float *)execution.pipes[scoresPipe];
    float *input = (float *)execution.pipes[inputPipe];
    for (NnUint batch = 0u; batch < batchSize; ++batch) {
        scores[batch * nExperts] = 0.2f + 0.01f * (float)batch;
        scores[batch * nExperts + 1u] = 0.8f - 0.02f * (float)batch;
        for (NnUint d = 0u; d < dim; ++d)
            input[batch * dim + d] = (float)(10u * batch + d + 1u);
    }
    executor.forward();

    const float *output = (const float *)execution.pipes[outputPipe];
    for (NnUint batch = 0u; batch < batchSize; ++batch) {
        const float scale = scores[batch * nExperts] + 2.0f * scores[batch * nExperts + 1u];
        for (NnUint d = 0u; d < dim; ++d)
            assert(std::fabs(output[batch * dim + d] - input[batch * dim + d] * scale) < 0.0001f);
    }

    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

static void testDeviceBufferHandoff() {
    NnNetConfigBuilder netBuilder(1u, 1u);
    const NnUint pipe = netBuilder.addPipe("device_handoff", size2D(F_32, 1u, 4u));
    NnNodeConfigBuilder nodeBuilder(0u);
    const NnUint state = nodeBuilder.addBuffer("state", size2D(F_32, 1u, 4u));

    NnSegmentConfigBuilder writeSegment;
    writeSegment.addOp(OP_CAST, "write_state", 0u,
        pointerBatchConfig(SRC_PIPE, pipe), pointerBatchConfig(SRC_BUFFER, state),
        size0(), NnCastOpCodeConfig{});
    nodeBuilder.addSegment(writeSegment.build());

    NnSegmentConfigBuilder readSegment;
    readSegment.addOp(OP_SCALE, "read_state", 0u,
        pointerBatchConfig(SRC_BUFFER, state), pointerBatchConfig(SRC_PIPE, pipe),
        size0(), NnScaleOpCodeConfig{0u, 2.0f, 1u});
    nodeBuilder.addSegment(readSegment.build());

    NnNetConfig net = netBuilder.build();
    NnNodeConfig node = nodeBuilder.build();
    NnNetExecution execution(1u, &net);
    execution.setBatchSize(1u);
    std::vector<NnExecutorDevice> devices;
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&net, &node, &execution), 0, 0));
    devices.push_back(NnExecutorDevice(new NnCpuDevice(&net, &node, &execution), 1, 1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&net, &node, &devices, &execution, &synchronizer, false);

    float *values = (float *)execution.pipes[pipe];
    for (NnUint i = 0u; i < 4u; ++i)
        values[i] = (float)(i + 1u);
    executor.forward();
    for (NnUint i = 0u; i < 4u; ++i)
        assert(std::fabs(values[i] - 2.0f * (float)(i + 1u)) < 0.000001f);

    for (NnUint i = 0u; i < 4u; ++i)
        values[i] = (float)(10u + i);
    executor.forward(0u, 0u);
    executor.forward(1u, 1u);
    for (NnUint i = 0u; i < 4u; ++i)
        assert(std::fabs(values[i] - 2.0f * (float)(10u + i)) < 0.000001f);

    devices.clear();
    releaseNodeConfig(&node);
    releaseNetConfig(&net);
}

int main() {
    initQuants();
    testSharedWeights();
    testWeightMulAndImmediateScale();
    testWeightedExpertScale();
    testPackedGroupedReduction();
    testPackedGroupedQuantizedReduction();
    testPackedExpertMatmulAndMerge();
    testDeviceBufferHandoff();

    NnNetConfig netConfig;
    NnNodeConfig nodeConfig;
    NnNetExecution *execution;
    NnFakeNodeSynchronizer *synchronizer;
    std::vector<NnExecutorDevice> devices;
    NnExecutor *executor = buildExecutor(
        &netConfig, &nodeConfig, &execution, &synchronizer, &devices);

    resetInput(execution, 1.0f);
    executor->forward(0u, 0u);
    assert(std::fabs(((float *)execution->pipes[0])[0] - std::tanh(1.0f)) < 0.000001f);

    resetInput(execution, 1.0f);
    executor->forward(1u, 1u);
    const float expectedSilu = 1.0f / (1.0f + std::exp(-1.0f));
    assert(std::fabs(((float *)execution->pipes[0])[0] - expectedSilu) < 0.001f);

    resetInput(execution, 1.0f);
    executor->forward();
    const float expectedAll = std::tanh(1.0f) / (1.0f + std::exp(-std::tanh(1.0f)));
    assert(std::fabs(((float *)execution->pipes[0])[0] - expectedAll) < 0.001f);

    delete executor;
    devices.clear();
    delete synchronizer;
    delete execution;
    releaseNodeConfig(&nodeConfig);
    releaseNetConfig(&netConfig);
    return 0;
}
