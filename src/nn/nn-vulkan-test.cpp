#include <cstdio>
#include <cmath>
#include "nn-config-builder.hpp"
#include "nn-quants.hpp"
#include "nn-vulkan.hpp"

#define N_BATCHES 4

void printOk(const char *name) {
    printf("✅ %24s passed\n", name);
}

void assertFloat(NnUint position, const float value, const float expectedValue, const float tolerance) {
    float diff = fabs(expectedValue - value);
    if (!std::isfinite(value) || !std::isfinite(expectedValue) || diff > tolerance) {
        printf("❌ [%d] failed: value=%f, expectedValue=%f, diff=%f\n", position, value, expectedValue, diff);
        exit(1);
    }
}

void execute(
    void (*build)(NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder),
    void (*execute)(NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device),
    NnUint nBatches = N_BATCHES
) {
    NnUint nNodes = 1;
    NnNetConfigBuilder netBuilder(nNodes, nBatches);
    NnNodeConfigBuilder nodeBuilder(0);
    NnSegmentConfigBuilder segmentBuilder;
    build(&netBuilder, &nodeBuilder, &segmentBuilder);
    nodeBuilder.addSegment(segmentBuilder.build());

    NnNetConfig netConfig = netBuilder.build();
    NnNodeConfig nodeConfig = nodeBuilder.build();
    std::unique_ptr<NnNetConfig, void(*)(NnNetConfig *)> netConfigPtr(&netConfig, releaseNetConfig);
    std::unique_ptr<NnNodeConfig, void(*)(NnNodeConfig *)> nodeConfigPtr(&nodeConfig, releaseNodeConfig);

    NnNetExecution execution(1, &netConfig);

    NnUint gpuIndex = 0;
    std::vector<NnExecutorDevice> devices;
    NnVulkanDevice *device = new NnVulkanDevice(gpuIndex, &netConfig, &nodeConfig, &execution);
    devices.push_back(NnExecutorDevice(device, -1, -1));
    NnFakeNodeSynchronizer synchronizer;
    NnExecutor executor(&netConfig, &nodeConfig, &devices, &execution, &synchronizer, false);

    execute(&executor, &execution, device);
}

template <NnUint dim>
void testRmsNorm_F32_F32_F32() {
    #define TEST_RMS_NORM_EPS 1e-5f
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1));
            segmentBuilder->addOp(OP_INV_RMS, "inv_rms", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{TEST_RMS_NORM_EPS, 1});
            segmentBuilder->addOp(OP_RMS_NORM, "rms_norm", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size1D(F_32, dim),
                NnRmsNormOpConfig{invRmsBufferIndex, 1});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            const NnUint batchSize = 2;
            execution->setBatchSize(batchSize);

            std::vector<float> normWeight(dim);
            for (NnUint i = 0; i < dim; i++)
                normWeight[i] = (0.25f + (float)i) / (float)dim;
            executor->loadWeight("rms_norm", 0u, 0u, normWeight.size() * sizeof(float), (NnByte *)normWeight.data());

            float *xPipe = (float *)execution->pipes[0];
            float expectedS[batchSize];
            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * dim];
                float s = 0.0f;
                for (NnUint i = 0; i < dim; i++) {
                    float u = (float)(dim - i + b) / (float)(dim / 2);
                    xBatchPipe[i] = u;
                    s += u * u;
                }
                s /= (float)dim;
                expectedS[b] = 1.0f / sqrtf(s + TEST_RMS_NORM_EPS);
            }

            // act
            executor->forward();

            // assert
            float invRmsBuffer[N_BATCHES];
            device->data.buffers[0].get()->read((NnByte *)invRmsBuffer);

            for (NnUint b = 0; b < batchSize; b++) {
                float *xBatchPipe = &xPipe[b * dim];

                const float t = 0.0000019f;
                assertFloat(b, invRmsBuffer[b], expectedS[b], t);
                const float s = invRmsBuffer[b];
                for (NnUint i = 0; i < dim; i++) {
                    float u = (float)(dim - i + b) / (float)(dim / 2);
                    assertFloat(b * dim + i, xBatchPipe[i], (u * s) * normWeight[i], t);
                }
            }
            printOk("testRmsNorm_F32_F32_F32");
        });
}

template <NnUint dim>
void testSilu_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_SILU, "silu", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSiluOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float expectedOutput[dim * N_BATCHES];
            float *xPipe = (float *)execution->pipes[0];
    
            for (NnUint b = 0; b < N_BATCHES; b++) {
                const NnUint offset = b * dim;
                for (NnUint i = 0; i < dim; i++) {
                    const float v = i / (float)dim + (float)(b + 1);
                    xPipe[offset + i] = v;
                    expectedOutput[offset + i] = v / (1.0 + expf(-v));
                }
            }

            // act
            executor->forward();

            // assert
            float t = 0.00001f;
            for (NnUint b = 0; b < N_BATCHES; b++) {
                const NnUint offset = b * dim;
                for (NnUint i = 0; i < dim; i++)
                    assertFloat(offset + i, xPipe[offset + i], expectedOutput[offset + i], t);
            }

            printOk("testSilu_F32_F32");
        });
}

template <NnUint dim>
void testRmsNormScaleFree_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint invRmsBufferIndex = nodeBuilder->addBuffer("inv_rms", size2D(F_32, N_BATCHES, 1u));
            segmentBuilder->addOp(OP_INV_RMS, "inv_rms", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_BUFFER, invRmsBufferIndex),
                size0(),
                NnInvRmsOpConfig{1e-5f, 1u});
            segmentBuilder->addOp(OP_RMS_NORM, "rms_norm_scale_free", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnRmsNormOpConfig{invRmsBufferIndex, 1u, 1u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float expected[dim * N_BATCHES];

            for (NnUint b = 0u; b < N_BATCHES; b++) {
                float sum = 0.0f;
                for (NnUint i = 0u; i < dim; i++) {
                    const float value = (float)(b * dim + i + 1u) / (float)dim;
                    xPipe[b * dim + i] = value;
                    sum += value * value;
                }
                const float invRms = 1.0f / sqrtf(sum / (float)dim + 1e-5f);
                for (NnUint i = 0u; i < dim; i++)
                    expected[b * dim + i] = xPipe[b * dim + i] * invRms;
            }

            executor->forward();

            for (NnUint i = 0u; i < dim * N_BATCHES; i++)
                assertFloat(i, xPipe[i], expected[i], 0.00001f);
            printOk("testRmsNormScaleFree_F32_F32");
        });
}

template <NnUint dim>
void testTanh_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_TANH, "tanh", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnTanhOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float expectedOutput[dim * N_BATCHES];

            for (NnUint i = 0u; i < dim * N_BATCHES; i++) {
                xPipe[i] = ((float)i - (float)(dim * N_BATCHES / 2u)) / (float)dim;
                expectedOutput[i] = tanhf(xPipe[i]);
            }

            executor->forward();

            for (NnUint i = 0u; i < dim * N_BATCHES; i++)
                assertFloat(i, xPipe[i], expectedOutput[i], 0.00001f);
            printOk("testTanh_F32_F32");
        });
}

template <NnUint dim>
void testGelu_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_GELU, "gelu", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSiluOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float expected[dim * N_BATCHES];
            const float coefficient = 0.7978845608028654f;
            for (NnUint i = 0u; i < dim * N_BATCHES; ++i) {
                const float value = ((float)i - (float)(dim * N_BATCHES / 2u)) / (float)dim;
                xPipe[i] = value;
                expected[i] = 0.5f * value *
                    (1.0f + std::tanh(coefficient * (value + 0.044715f * value * value * value)));
            }
            executor->forward();
            for (NnUint i = 0u; i < dim * N_BATCHES; ++i)
                assertFloat(i, xPipe[i], expected[i], 0.00001f);
            printOk("testGelu_F32_F32");
        });
}

template <NnUint dim, NnUint nZ>
void testMul_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            NnUint sBufferIndex = nodeBuilder->addBuffer("s", size3D(F_32, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(OP_MUL, "mul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMulOpCodeConfig{sBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float sBuffer[nZ * N_BATCHES * dim];
            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++) {
                xPipe[i] = (float)i;
                sBuffer[i] = (i % 8) / 10.0f;
            }

            device->data.buffers[0].get()->write((NnByte *)sBuffer);

            // act
            executor->forward();

            // assert
            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                assertFloat(i, xPipe[i], i * ((i % 8) / 10.0f), 0.000001f);
            printOk("testMul_F32_F32");
        });
}

void testMergeAdd_F32_F32() {
    #define MERGE_ADD_F32_NODES 2
    #define MERGE_ADD_F32_DIM 64
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint zPipeIndex = netBuilder->addPipe("Z", size2D(F_32, N_BATCHES, MERGE_ADD_F32_DIM * MERGE_ADD_F32_NODES));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MERGE_ADD_F32_DIM));
            segmentBuilder->addOp(OP_MERGE_ADD, "mergeAdd", 0,
                pointerBatchConfig(SRC_PIPE, zPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *zPipe = (float *)execution->pipes[0];
            float *xPipe = (float *)execution->pipes[1];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint n = 0; n < MERGE_ADD_F32_NODES; n++) {
                    for (NnUint i = 0; i < MERGE_ADD_F32_DIM; i++)
                        zPipe[b * MERGE_ADD_F32_NODES * MERGE_ADD_F32_DIM + n * MERGE_ADD_F32_DIM + i] = (float)(b + 1);
                }
            }

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < MERGE_ADD_F32_DIM; i++) {
                    NnUint j = b * MERGE_ADD_F32_DIM + i;
                    assertFloat(j, xPipe[j], (float)(2 * b + 2), 0.00001f);
                }
            }
            printOk("testMergeAdd_F32_F32");
        });
}

void testMergeSum_F32_F32() {
    #define MERGE_SUM_F32_N_Z 2
    #define MERGE_SUM_F32_DIM 4
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, MERGE_SUM_F32_N_Z, N_BATCHES, MERGE_SUM_F32_DIM));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_32, 1u, N_BATCHES, MERGE_SUM_F32_DIM));
            segmentBuilder->addOp(OP_MERGE_SUM, "merge_sum", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnMergeSumOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(2);

            float *xPipe = (float *)execution->pipes[0];
            float *xPipeZ0 = &xPipe[0];
            float *xPipeZ1 = &xPipe[N_BATCHES * MERGE_SUM_F32_DIM];
            float *yPipe = (float *)execution->pipes[1];
            xPipeZ0[0] = 1.0f;
            xPipeZ0[1] = 1.1f;
            xPipeZ0[2] = 1.2f;
            xPipeZ0[3] = 1.3f;

            xPipeZ0[4] = 2.0f;
            xPipeZ0[5] = 2.1f;
            xPipeZ0[6] = 2.2f;
            xPipeZ0[7] = 2.3f;

            xPipeZ1[0] = 0.5f;
            xPipeZ1[1] = 0.1f;
            xPipeZ1[2] = 0.2f;
            xPipeZ1[3] = 0.3f;

            xPipeZ1[4] = 0.4f;
            xPipeZ1[5] = 0.3f;
            xPipeZ1[6] = 0.2f;
            xPipeZ1[7] = 0.1f;

            // act
            executor->forward();

            const float t = 0.00001f;
            assertFloat(0, yPipe[0], 1.5f, t);
            assertFloat(1, yPipe[1], 1.2f, t);
            assertFloat(2, yPipe[2], 1.4f, t);
            assertFloat(3, yPipe[3], 1.6f, t);

            assertFloat(4, yPipe[4], 2.4f, t);
            assertFloat(5, yPipe[5], 2.4f, t);
            assertFloat(6, yPipe[6], 2.4f, t);
            assertFloat(7, yPipe[7], 2.4f, t);

            printOk("testMergeSum_F32_F32");
        });
}

template <NnUint nNodes, NnUint dim>
static void testMergeAdd_Q80_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint zPipeIndex = netBuilder->addPipe("Z", size2D(F_Q80, N_BATCHES, dim * nNodes));
            const NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            segmentBuilder->addOp(OP_MERGE_ADD, "mergeAdd", 0,
                pointerBatchConfig(SRC_PIPE, zPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMergeAddOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float z[N_BATCHES * dim * nNodes];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint n = 0; n < nNodes; n++) {
                    for (NnUint i = 0; i < dim; i++)
                        z[b * nNodes * dim + n * dim + i] = (float)(b + 1);
                }
            }

            NnBlockQ80 *zPipe = (NnBlockQ80 *)execution->pipes[0];
            const float *xPipe = (float *)execution->pipes[1];
            quantizeF32toQ80(z, zPipe, N_BATCHES * dim * nNodes, 1, 0);

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < dim; i++) {
                    float expectedValue = (float)((b + 1) * nNodes);
                    NnUint j = b * dim + i;
                    assertFloat(j, xPipe[j], expectedValue, 0.001f);
                }
            }
            printOk("testMergeAdd_Q80_F32");
        });
}

void testEmbedding_F32_F32() {
    #define EMBEDDING_DIM 16
    #define EMBEDDING_LEN 8
    assert(EMBEDDING_LEN > N_BATCHES);
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, EMBEDDING_DIM));
            segmentBuilder->addOp(OP_EMBEDDING, "embedding", 0,
                pointerBatchConfig(SRC_PIPE, posPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size2D(F_32, EMBEDDING_LEN, EMBEDDING_DIM),
                NnEmbeddingOpConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float embedding[EMBEDDING_DIM * EMBEDDING_LEN];
            for (NnUint l = 0; l < EMBEDDING_LEN; l++) {
                for (NnUint i = 0; i < EMBEDDING_DIM; i++)
                    embedding[l * EMBEDDING_DIM + i] = (float)(l + 4);
            }
            float *posPipe = (float *)execution->pipes[0];
            for (NnUint b = 0; b < N_BATCHES; b++)
                posPipe[b] = (float)b;

            executor->loadWeight("embedding", 0u, 0u, EMBEDDING_DIM * EMBEDDING_LEN * sizeof(float), (NnByte *)embedding);

            // act
            executor->forward();

            // assert
            float *xPipe = (float *)execution->pipes[1];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < EMBEDDING_DIM; i++) {
                    NnUint j = b * EMBEDDING_DIM + i;
                    assertFloat(j, xPipe[j], (float)(b + 4), 0.00001f);
                }
            }

            printOk("testEmbedding_F32_F32");
        });
}

template <bool quantized>
void testShardedEmbedding_F32() {
    #define SHARDED_EMBEDDING_DIM 32
    #define SHARDED_EMBEDDING_VOCAB 4
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint tokenPipeIndex = netBuilder->addPipe("tokens", size2D(F_32, N_BATCHES, 1u));
            const NnUint outputPipeIndex = netBuilder->addPipe("embedding",
                size2D(F_32, N_BATCHES, SHARDED_EMBEDDING_DIM));
            segmentBuilder->addOp(OP_EMBEDDING, "sharded_embedding", 0u,
                pointerBatchConfig(SRC_PIPE, tokenPipeIndex),
                pointerBatchConfig(SRC_PIPE, outputPipeIndex),
                size2D(quantized ? F_Q40 : F_32, SHARDED_EMBEDDING_VOCAB, SHARDED_EMBEDDING_DIM),
                NnEmbeddingOpConfig{10u, SHARDED_EMBEDDING_VOCAB, 2.0f});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            float *tokens = (float *)execution->pipes[0];
            tokens[0] = 9.0f;
            tokens[1] = 10.0f;
            tokens[2] = 13.0f;
            tokens[3] = 14.0f;
            float embedding[SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM];
            for (NnUint row = 0u; row < SHARDED_EMBEDDING_VOCAB; ++row)
                for (NnUint h = 0u; h < SHARDED_EMBEDDING_DIM; ++h)
                    embedding[row * SHARDED_EMBEDDING_DIM + h] = (float)(row + 1u) + (float)h * 0.01f;
            float effective[SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM];
            if (quantized) {
                NnBlockQ40 packed[SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM / Q40_BLOCK_SIZE];
                quantizeF32toQ40(embedding, packed, SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM, 1u, 0u);
                dequantizeQ40toF32(packed, effective,
                    SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM, 1u, 0u);
                executor->loadWeight("sharded_embedding", 0u, 0u, sizeof(packed), (NnByte *)packed);
            } else {
                std::copy(embedding, embedding + SHARDED_EMBEDDING_VOCAB * SHARDED_EMBEDDING_DIM, effective);
                executor->loadWeight("sharded_embedding", 0u, 0u, sizeof(embedding), (NnByte *)embedding);
            }
            executor->forward();
            const float *output = (const float *)execution->pipes[1];
            for (NnUint h = 0u; h < SHARDED_EMBEDDING_DIM; ++h) {
                assertFloat(h, output[h], 0.0f, 0.00001f);
                assertFloat(SHARDED_EMBEDDING_DIM + h, output[SHARDED_EMBEDDING_DIM + h],
                    2.0f * effective[h], 0.0001f);
                assertFloat(2u * SHARDED_EMBEDDING_DIM + h, output[2u * SHARDED_EMBEDDING_DIM + h],
                    2.0f * effective[3u * SHARDED_EMBEDDING_DIM + h], 0.0001f);
                assertFloat(3u * SHARDED_EMBEDDING_DIM + h, output[3u * SHARDED_EMBEDDING_DIM + h],
                    0.0f, 0.00001f);
            }
            printOk(quantized ? "testShardedEmbedding_Q40" : "testShardedEmbedding_F32");
        });
}

void testMulWeight_F32_F32() {
    #define MUL_WEIGHT_DIM 32
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint inputPipeIndex = netBuilder->addPipe("input", size2D(F_32, N_BATCHES, MUL_WEIGHT_DIM));
            const NnUint outputPipeIndex = netBuilder->addPipe("output", size2D(F_32, N_BATCHES, MUL_WEIGHT_DIM));
            segmentBuilder->addOp(OP_MUL, "mul_weight", 0u,
                pointerBatchConfig(SRC_PIPE, inputPipeIndex), pointerBatchConfig(SRC_PIPE, outputPipeIndex),
                size1D(F_32, MUL_WEIGHT_DIM), NnMulOpCodeConfig{0u, 1u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            float *input = (float *)execution->pipes[0];
            float weight[MUL_WEIGHT_DIM];
            for (NnUint h = 0u; h < MUL_WEIGHT_DIM; ++h)
                weight[h] = 0.5f + (float)h * 0.01f;
            for (NnUint i = 0u; i < N_BATCHES * MUL_WEIGHT_DIM; ++i)
                input[i] = (float)(i + 1u) * 0.1f;
            executor->loadWeight("mul_weight", 0u, 0u, sizeof(weight), (NnByte *)weight);
            executor->forward();
            const float *output = (const float *)execution->pipes[1];
            for (NnUint batch = 0u; batch < N_BATCHES; ++batch)
                for (NnUint h = 0u; h < MUL_WEIGHT_DIM; ++h) {
                    const NnUint index = batch * MUL_WEIGHT_DIM + h;
                    assertFloat(index, output[index], input[index] * weight[h], 0.00001f);
                }
            printOk("testMulWeight_F32_F32");
        });
}

void testRepeatZ_F32_F32() {
    #define REPEAT_Z_DIM 8
    #define REPEAT_Z_COUNT 3
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint inputPipeIndex = netBuilder->addPipe("input", size2D(F_32, N_BATCHES, REPEAT_Z_DIM));
            const NnUint outputPipeIndex = netBuilder->addPipe("output",
                size3D(F_32, REPEAT_Z_COUNT, N_BATCHES, REPEAT_Z_DIM));
            segmentBuilder->addOp(OP_REPEAT_Z, "repeat_z", 0u,
                pointerBatchConfig(SRC_PIPE, inputPipeIndex), pointerBatchConfig(SRC_PIPE, outputPipeIndex),
                size0(), NnRepeatZOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            float *input = (float *)execution->pipes[0];
            for (NnUint i = 0u; i < N_BATCHES * REPEAT_Z_DIM; ++i)
                input[i] = (float)(i + 1u);
            executor->forward();
            const float *output = (const float *)execution->pipes[1];
            for (NnUint z = 0u; z < REPEAT_Z_COUNT; ++z)
                for (NnUint i = 0u; i < N_BATCHES * REPEAT_Z_DIM; ++i)
                    assertFloat(z * N_BATCHES * REPEAT_Z_DIM + i,
                        output[z * N_BATCHES * REPEAT_Z_DIM + i], input[i], 0.00001f);
            printOk("testRepeatZ_F32_F32");
        });
}

template <bool quantized>
void testGroupedMergeSum_F32() {
    #define GROUPED_MERGE_DIM 32
    #define GROUPED_MERGE_GROUPS 2
    #define GROUPED_MERGE_CONTRIBUTIONS 3
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint inputPipeIndex = netBuilder->addPipe("input",
                size2D(quantized ? F_Q80 : F_32, N_BATCHES,
                    GROUPED_MERGE_CONTRIBUTIONS * GROUPED_MERGE_GROUPS * GROUPED_MERGE_DIM));
            const NnUint outputPipeIndex = netBuilder->addPipe("output",
                size3D(F_32, GROUPED_MERGE_GROUPS, N_BATCHES, GROUPED_MERGE_DIM));
            segmentBuilder->addOp(OP_MERGE_SUM, "grouped_merge_sum", 0u,
                pointerBatchConfig(SRC_PIPE, inputPipeIndex), pointerBatchConfig(SRC_PIPE, outputPipeIndex),
                size0(), NnMergeSumOpCodeConfig{GROUPED_MERGE_GROUPS});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            const NnUint rowSize = GROUPED_MERGE_CONTRIBUTIONS * GROUPED_MERGE_GROUPS * GROUPED_MERGE_DIM;
            float inputValues[N_BATCHES * rowSize];
            for (NnUint batch = 0u; batch < N_BATCHES; ++batch)
                for (NnUint contribution = 0u; contribution < GROUPED_MERGE_CONTRIBUTIONS; ++contribution)
                    for (NnUint group = 0u; group < GROUPED_MERGE_GROUPS; ++group)
                        for (NnUint h = 0u; h < GROUPED_MERGE_DIM; ++h) {
                            const NnUint index = batch * rowSize +
                                (contribution * GROUPED_MERGE_GROUPS + group) * GROUPED_MERGE_DIM + h;
                            inputValues[index] = (float)(100u * batch + 10u * contribution + group) +
                                (float)h * 0.001f;
                        }
            if (quantized) {
                quantizeF32toQ80(inputValues, (NnBlockQ80 *)execution->pipes[0],
                    N_BATCHES * rowSize, 1u, 0u);
                dequantizeQ80toF32((const NnBlockQ80 *)execution->pipes[0], inputValues,
                    N_BATCHES * rowSize, 1u, 0u);
            } else {
                std::copy(inputValues, inputValues + N_BATCHES * rowSize,
                    (float *)execution->pipes[0]);
            }
            executor->forward();
            const float *output = (const float *)execution->pipes[1];
            for (NnUint group = 0u; group < GROUPED_MERGE_GROUPS; ++group)
                for (NnUint batch = 0u; batch < N_BATCHES; ++batch)
                    for (NnUint h = 0u; h < GROUPED_MERGE_DIM; ++h) {
                        float expected = 0.0f;
                        for (NnUint contribution = 0u; contribution < GROUPED_MERGE_CONTRIBUTIONS; ++contribution)
                            expected += inputValues[batch * rowSize +
                                (contribution * GROUPED_MERGE_GROUPS + group) * GROUPED_MERGE_DIM + h];
                        const NnUint index = (group * N_BATCHES + batch) * GROUPED_MERGE_DIM + h;
                        assertFloat(index, output[index], expected, 0.0001f);
                    }
            printOk(quantized ? "testGroupedMergeSum_Q80" : "testGroupedMergeSum_F32");
        });
}

#define EMBEDDING_MIX_VOCAB 4
#define EMBEDDING_MIX_HIDDEN 32
#define EMBEDDING_MIX_PHYSICAL 64

template <bool useDivisor>
void testReduceMax_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint logitsPipeIndex = netBuilder->addPipe("logits",
                size2D(F_32, N_BATCHES, EMBEDDING_MIX_VOCAB));
            const NnUint temperaturePipeIndex = netBuilder->addPipe("temperature", size1D(F_32, 1u));
            const NnUint maximumPipeIndex = netBuilder->addPipe("maximum", size2D(F_32, N_BATCHES, 1u));
            netBuilder->addPreSync(temperaturePipeIndex);
            segmentBuilder->addOp(OP_REDUCE_MAX, "embedding_mix_max", 0u,
                pointerBatchConfig(SRC_PIPE, logitsPipeIndex),
                pointerBatchConfig(SRC_PIPE, maximumPipeIndex), size0(),
                NnReduceMaxOpConfig{temperaturePipeIndex, useDivisor ? 1u : 0u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(2u);
            float *logits = (float *)execution->pipes[0];
            const float values[] = {1.0f, 4.0f, 2.0f, 3.0f, -2.0f, -5.0f, 0.0f, -1.0f};
            std::copy(values, values + 8u, logits);
            ((float *)execution->pipes[1])[0] = 2.0f;
            executor->forward();
            const float *maximum = (const float *)execution->pipes[2];
            assertFloat(0u, maximum[0], useDivisor ? 2.0f : 4.0f, 0.00001f);
            assertFloat(1u, maximum[1], 0.0f, 0.00001f);
            printOk(useDivisor ? "testReduceMaxDivisor_F32" : "testReduceMax_F32");
        });
}

void testEmbeddingMixRescale_F32_F32() {
    constexpr NnUint width = 4u;
    constexpr NnUint nNodes = 2u;
    constexpr NnUint nodeIndex = 1u;
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder,
            NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint localMaximumPipeIndex = netBuilder->addPipe("local_maximum",
                size2D(F_32, N_BATCHES, nNodes));
            netBuilder->addPreSync(localMaximumPipeIndex);
            const NnUint mixBufferIndex = nodeBuilder->addBuffer("mix",
                size2D(F_32, N_BATCHES, width));
            const NnUint globalMaximumBufferIndex = nodeBuilder->addBuffer("global_maximum",
                size2D(F_32, N_BATCHES, 1u));
            segmentBuilder->addOp(OP_EMBEDDING_MIX_RESCALE, "embedding_mix_rescale", 0u,
                pointerBatchConfig(SRC_BUFFER, mixBufferIndex),
                pointerBatchConfig(SRC_BUFFER, mixBufferIndex), size0(),
                NnEmbeddingMixRescaleOpConfig{localMaximumPipeIndex,
                    globalMaximumBufferIndex, nodeIndex, nNodes});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            constexpr NnUint batchSize = 2u;
            execution->setBatchSize(batchSize);
            const float localMaximum[] = {2.0f, 1.0f, -1.0f, 3.0f};
            std::copy(localMaximum, localMaximum + batchSize * nNodes,
                (float *)execution->pipes[0]);
            const float input[N_BATCHES * width] = {
                1.0f, 2.0f, 3.0f, 4.0f, -1.0f, -2.0f, -3.0f, -4.0f};
            const float globalMaximum[N_BATCHES] = {2.0f, 4.0f};
            device->writeBuffer(0u, (const NnByte *)input);
            device->writeBuffer(1u, (const NnByte *)globalMaximum);
            executor->forward();
            float output[N_BATCHES * width];
            device->readBuffer(0u, (NnByte *)output);
            const float scale = expf(-1.0f);
            for (NnUint i = 0u; i < batchSize * width; ++i)
                assertFloat(i, output[i], input[i] * scale, 0.00001f);
            printOk("testEmbeddingMixRescale_F32");
        });
}

template <bool quantized, NnUint vocabSize = EMBEDDING_MIX_VOCAB,
    NnUint hiddenSize = EMBEDDING_MIX_HIDDEN, NnUint tileSize = 2u,
    NnUint batchSize = 2u, NnUint capacity = N_BATCHES,
    NnUint physicalSize = ((hiddenSize + 2u + 31u) / 32u) * 32u>
void testTiedEmbedding_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint tileCount = (vocabSize + tileSize - 1u) / tileSize;
            const NnUint hiddenPipeIndex = netBuilder->addPipe("hidden",
                size2D(quantized ? F_Q80 : F_32, capacity, hiddenSize));
            const NnUint temperaturePipeIndex = netBuilder->addPipe("temperature", size1D(F_32, 1u));
            const NnUint logitsPipeIndex = netBuilder->addPipe("logits",
                size2D(F_32, capacity, vocabSize));
            const NnUint maximumPipeIndex = netBuilder->addPipe("maximum",
                size2D(F_32, capacity, 1u));
            const NnUint outputPipeIndex = netBuilder->addPipe("output",
                size2D(F_32, capacity, hiddenSize));
            netBuilder->addPreSync(hiddenPipeIndex);
            netBuilder->addPreSync(temperaturePipeIndex);
            const NnUint partialBufferIndex = nodeBuilder->addBuffer("partials",
                size2D(F_32, capacity, tileCount * physicalSize));
            const NnUint probabilityBufferIndex = nodeBuilder->addBuffer("probabilities",
                size2D(F_32, capacity, vocabSize));
            const NnUint matmulProbabilityBufferIndex = quantized
                ? nodeBuilder->addBuffer("quant_probabilities",
                    size2D(F_Q80, capacity, vocabSize))
                : probabilityBufferIndex;
            const NnUint numeratorBufferIndex = nodeBuilder->addBuffer("numerator",
                size2D(F_32, capacity, hiddenSize));
            const NnUint localMixBufferIndex = nodeBuilder->addBuffer("local_mix",
                size2D(F_32, capacity, physicalSize));
            segmentBuilder->addOp(OP_MATMUL, "tied_embedding_weight", 0u,
                pointerBatchConfig(SRC_PIPE, hiddenPipeIndex),
                pointerBatchConfig(SRC_PIPE, logitsPipeIndex),
                size2D(quantized ? F_Q40 : F_32, hiddenSize, vocabSize),
                NnMatmulOpConfig{0u, 0u, 0u});
            segmentBuilder->addOp(OP_TIED_EMBEDDING_PARTIAL,
                "tied_embedding_probabilities", 0u,
                pointerBatchConfig(SRC_PIPE, logitsPipeIndex),
                pointerBatchConfig(SRC_PIPE, logitsPipeIndex),
                size0(),
                NnTiedEmbeddingPartialOpConfig{temperaturePipeIndex,
                    partialBufferIndex, probabilityBufferIndex, physicalSize,
                    tileSize, tileCount, hiddenSize, 2.5f});
            if (quantized) {
                segmentBuilder->addOp(OP_CAST, "tied_embedding_quantize_probabilities", 0u,
                    pointerBatchConfig(SRC_BUFFER, probabilityBufferIndex),
                    pointerBatchConfig(SRC_BUFFER, matmulProbabilityBufferIndex),
                    size0(), NnCastOpCodeConfig{});
            }
            segmentBuilder->addOp(OP_MATMUL, "tied_embedding_mix_weight", 0u,
                pointerBatchConfig(SRC_BUFFER, matmulProbabilityBufferIndex),
                pointerBatchConfig(SRC_BUFFER, numeratorBufferIndex),
                size2D(quantized ? F_Q40 : F_32, vocabSize, hiddenSize),
                NnMatmulOpConfig{0u, 0u, partialBufferIndex});
            segmentBuilder->addOp(OP_TIED_EMBEDDING_REDUCE, "tied_embedding_reduce", 0u,
                pointerBatchConfig(SRC_BUFFER, partialBufferIndex),
                pointerBatchConfig(SRC_PIPE, maximumPipeIndex), size0(),
                NnTiedEmbeddingReduceOpConfig{localMixBufferIndex, numeratorBufferIndex,
                    physicalSize, tileCount, hiddenSize});
            segmentBuilder->addOp(OP_EMBEDDING_MIX_FINALIZE, "tied_embedding_finalize", 0u,
                pointerBatchConfig(SRC_BUFFER, localMixBufferIndex),
                pointerBatchConfig(SRC_PIPE, outputPipeIndex), size0(),
                NnEmbeddingMixFinalizeOpConfig{physicalSize, 2.0f});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(batchSize);
            const float temperature = 0.5f;
            ((float *)execution->pipes[1])[0] = temperature;

            std::vector<float> hidden(batchSize * hiddenSize);
            std::vector<float> effectiveHidden(batchSize * hiddenSize);
            for (NnUint batch = 0u; batch < batchSize; ++batch)
                for (NnUint h = 0u; h < hiddenSize; ++h)
                    hidden[batch * hiddenSize + h] =
                        float(int((batch + 2u) * (h + 3u) % 17u) - 8) * 0.025f;
            if (quantized) {
                NnBlockQ80 *packed = (NnBlockQ80 *)execution->pipes[0];
                quantizeF32toQ80(hidden.data(), packed, batchSize * hiddenSize, 1u, 0u);
                dequantizeQ80toF32(packed, effectiveHidden.data(),
                    batchSize * hiddenSize, 1u, 0u);
            } else {
                std::copy(hidden.begin(), hidden.end(),
                    (float *)execution->pipes[0]);
                effectiveHidden = hidden;
            }

            std::vector<float> embedding(vocabSize * hiddenSize);
            for (NnUint token = 0u; token < vocabSize; ++token)
                for (NnUint h = 0u; h < hiddenSize; ++h)
                    embedding[token * hiddenSize + h] =
                        float(int((token + 1u) * (h + 5u) % 19u) - 9) * 0.02f;
            std::vector<float> effectiveEmbedding(vocabSize * hiddenSize);
            std::vector<float> effectiveMixEmbedding(vocabSize * hiddenSize);
            std::vector<float> transposed(vocabSize * hiddenSize);
            for (NnUint token = 0u; token < vocabSize; ++token)
                for (NnUint h = 0u; h < hiddenSize; ++h)
                    transposed[h * vocabSize + token] =
                        embedding[token * hiddenSize + h];
            if (quantized) {
                std::vector<NnBlockQ40> packed(vocabSize * hiddenSize / Q40_BLOCK_SIZE);
                quantizeF32toQ40(embedding.data(), packed.data(),
                    vocabSize * hiddenSize, 1u, 0u);
                dequantizeQ40toF32(packed.data(), effectiveEmbedding.data(),
                    vocabSize * hiddenSize, 1u, 0u);
                executor->loadWeight("tied_embedding_weight", 0u, 0u,
                    packed.size() * sizeof(NnBlockQ40), (NnByte *)packed.data());
                std::vector<float> effectiveTransposed(vocabSize * hiddenSize);
                quantizeF32toQ40(transposed.data(), packed.data(), transposed.size(), 1u, 0u);
                dequantizeQ40toF32(packed.data(), effectiveTransposed.data(),
                    effectiveTransposed.size(), 1u, 0u);
                for (NnUint token = 0u; token < vocabSize; ++token)
                    for (NnUint h = 0u; h < hiddenSize; ++h)
                        effectiveMixEmbedding[token * hiddenSize + h] =
                            effectiveTransposed[h * vocabSize + token];
                executor->loadWeight("tied_embedding_mix_weight", 0u, 0u,
                    packed.size() * sizeof(NnBlockQ40), (NnByte *)packed.data());
            } else {
                effectiveEmbedding = embedding;
                effectiveMixEmbedding = embedding;
                executor->loadWeight("tied_embedding_weight", 0u, 0u,
                    embedding.size() * sizeof(float), (NnByte *)embedding.data());
                executor->loadWeight("tied_embedding_mix_weight", 0u, 0u,
                    transposed.size() * sizeof(float), (NnByte *)transposed.data());
            }

            std::vector<float> expected(batchSize * hiddenSize);
            std::vector<float> expectedLogits(batchSize * vocabSize);
            for (NnUint batch = 0u; batch < batchSize; ++batch) {
                float maximum = -INFINITY;
                for (NnUint token = 0u; token < vocabSize; ++token) {
                    float raw = 0.0f;
                    for (NnUint h = 0u; h < hiddenSize; ++h)
                        raw += effectiveHidden[batch * hiddenSize + h] *
                            effectiveEmbedding[token * hiddenSize + h];
                    const float logit = 2.5f * tanhf(raw / 2.5f);
                    expectedLogits[batch * vocabSize + token] = logit;
                    maximum = std::max(maximum, logit / temperature);
                }
                float denominator = 0.0f;
                std::vector<float> probabilities(vocabSize);
                for (NnUint token = 0u; token < vocabSize; ++token) {
                    probabilities[token] = expf(
                        expectedLogits[batch * vocabSize + token] / temperature - maximum);
                    denominator += probabilities[token];
                }
                if (quantized) {
                    std::vector<NnBlockQ80> packedProbabilities(vocabSize / Q80_BLOCK_SIZE);
                    quantizeF32toQ80(probabilities.data(), packedProbabilities.data(),
                        vocabSize, 1u, 0u);
                    dequantizeQ80toF32(packedProbabilities.data(), probabilities.data(),
                        vocabSize, 1u, 0u);
                }
                for (NnUint h = 0u; h < hiddenSize; ++h) {
                    float numerator = 0.0f;
                    for (NnUint token = 0u; token < vocabSize; ++token)
                        numerator += probabilities[token] *
                            effectiveMixEmbedding[token * hiddenSize + h];
                    expected[batch * hiddenSize + h] = 2.0f * numerator / denominator;
                }
            }

            // Exercise both cached command buffers and runtime batch tails.
            const NnUint sizes[] = {batchSize, batchSize, 1u, batchSize};
            for (NnUint active : sizes) {
                execution->setBatchSize(active);
                executor->forward();
                const float *actualLogits = (const float *)execution->pipes[2];
                const float *actual = (const float *)execution->pipes[4];
                for (NnUint i = 0u; i < active * vocabSize; ++i)
                    assertFloat(i, actualLogits[i], expectedLogits[i], 0.0001f);
                for (NnUint i = 0u; i < active * hiddenSize; ++i)
                    assertFloat(i, actual[i], expected[i], quantized ? 0.001f : 0.0001f);
            }
            printOk(quantized ? "testTiedEmbedding_Q40" : "testTiedEmbedding_F32");
        }, capacity);
}

#define VOCAB_STATS_TEST_VOCAB 8
#define VOCAB_STATS_TEST_SHARDS 2

void testVocabOps_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint logitsPipeIndex = netBuilder->addPipe("logits",
                size2D(F_32, N_BATCHES, VOCAB_STATS_TEST_VOCAB));
            const NnUint temperaturePipeIndex = netBuilder->addPipe("temperature", size1D(F_32, 1u));
            const NnUint samplingCoinsPipeIndex = netBuilder->addPipe("sampling_coins",
                size2D(F_32, N_BATCHES, 1u));
            const NnUint targetTokensPipeIndex = netBuilder->addPipe("target_tokens",
                size2D(F_32, N_BATCHES, 1u));
            const NnUint sampledTokensPipeIndex = netBuilder->addPipe("sampled_tokens",
                size2D(F_32, N_BATCHES, 1u));
            netBuilder->addPreSync(temperaturePipeIndex);
            netBuilder->addPreSync(samplingCoinsPipeIndex);
            netBuilder->addPreSync(targetTokensPipeIndex);
            const NnUint globalMaxBufferIndex = nodeBuilder->addBuffer("global_max",
                size2D(F_32, N_BATCHES, 1u));
            const NnUint statsBufferIndex = nodeBuilder->addBuffer("stats",
                size2D(F_32, N_BATCHES, VOCAB_STATS_TEST_SHARDS * NN_VOCAB_STATS_WIDTH));
            segmentBuilder->addOp(OP_VOCAB_LOCAL_STATS, "vocab_local_stats", 0u,
                pointerBatchConfig(SRC_PIPE, logitsPipeIndex),
                pointerBatchConfig(SRC_BUFFER, statsBufferIndex), size0(),
                NnVocabLocalStatsOpConfig{globalMaxBufferIndex,
                    temperaturePipeIndex, targetTokensPipeIndex, 10u,
                    VOCAB_STATS_TEST_SHARDS, 1u});
            segmentBuilder->addOp(OP_VOCAB_SAMPLE, "vocab_sample", 0u,
                pointerBatchConfig(SRC_BUFFER, statsBufferIndex),
                pointerBatchConfig(SRC_PIPE, sampledTokensPipeIndex), size0(),
                NnVocabSampleOpConfig{globalMaxBufferIndex,
                    temperaturePipeIndex, samplingCoinsPipeIndex, logitsPipeIndex,
                    10u, 0u, VOCAB_STATS_TEST_SHARDS});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            const NnUint batchSize = 2u;
            execution->setBatchSize(batchSize);
            float *logits = (float *)execution->pipes[0];
            const float values[] = {
                0.0f, 0.2f, 0.4f, 0.6f, 0.8f, 1.0f, 1.2f, 1.4f,
                1.5f, 0.5f, -0.5f, -1.5f, -1.0f, 0.0f, 1.0f, 2.0f,
            };
            std::copy(values, values + 16u, logits);
            const float temperature = 0.75f;
            ((float *)execution->pipes[1])[0] = temperature;
            float *coins = (float *)execution->pipes[2];
            coins[0] = 0.0f;
            coins[1] = 0.999f;
            float *targets = (float *)execution->pipes[3];
            targets[0] = 12.0f;
            targets[1] = 15.0f;

            float maxima[N_BATCHES] = {0.0f};
            for (NnUint batch = 0u; batch < batchSize; ++batch) {
                maxima[batch] = logits[batch * VOCAB_STATS_TEST_VOCAB] / temperature;
                for (NnUint token = 1u; token < VOCAB_STATS_TEST_VOCAB; ++token)
                    maxima[batch] = std::max(maxima[batch],
                        logits[batch * VOCAB_STATS_TEST_VOCAB + token] / temperature);
            }
            device->data.buffers[0].get()->write((const NnByte *)maxima);

            float expectedSamples[batchSize];
            for (NnUint batch = 0u; batch < batchSize; ++batch) {
                float weights[VOCAB_STATS_TEST_VOCAB];
                float denominator = 0.0f;
                for (NnUint token = 0u; token < VOCAB_STATS_TEST_VOCAB; ++token) {
                    weights[token] = expf(
                        logits[batch * VOCAB_STATS_TEST_VOCAB + token] / temperature - maxima[batch]);
                    denominator += weights[token];
                }
                const float threshold = coins[batch] * denominator;
                float cumulative = 0.0f;
                NnUint sampled = VOCAB_STATS_TEST_VOCAB - 1u;
                for (NnUint token = 0u; token < VOCAB_STATS_TEST_VOCAB; ++token) {
                    cumulative += weights[token];
                    if (threshold < cumulative) {
                        sampled = token;
                        break;
                    }
                }
                expectedSamples[batch] = (float)(10u + sampled);
            }

            executor->forward();
            const float *actualSamples = (const float *)execution->pipes[4];
            for (NnUint batch = 0u; batch < batchSize; ++batch)
                assertFloat(batch, actualSamples[batch], expectedSamples[batch], 0.00001f);

            float stats[N_BATCHES * VOCAB_STATS_TEST_SHARDS * NN_VOCAB_STATS_WIDTH];
            device->data.buffers[1].get()->read((NnByte *)stats);
            assertFloat(4u, stats[4], logits[2], 0.00001f);
            const NnUint secondBatchOffset = VOCAB_STATS_TEST_SHARDS * NN_VOCAB_STATS_WIDTH;
            assert(std::isinf(stats[secondBatchOffset + 4u]) && stats[secondBatchOffset + 4u] < 0.0f);
            assertFloat(secondBatchOffset + 9u, stats[secondBatchOffset + 9u], logits[8u + 5u], 0.00001f);
            printOk("testVocabOps_F32_F32");
        });
}

#define DIFFUSION_ATTENTION_HEAD_DIM 4
#define DIFFUSION_ATTENTION_Q_HEADS 2
#define DIFFUSION_ATTENTION_KV_HEADS 1
#define DIFFUSION_ATTENTION_CACHE_CAPACITY 4

static void referenceDiffusionAttention(
    float *output,
    const float *query,
    const float *currentKey,
    const float *currentValue,
    const float *keyCache,
    const float *valueCache,
    const float *positions,
    const NnUint prefixLength,
    const NnUint queryLength,
    const NnDiffusionAttentionMode mode,
    const NnUint slidingWindow)
{
    const NnUint qDim = DIFFUSION_ATTENTION_Q_HEADS * DIFFUSION_ATTENTION_HEAD_DIM;
    const NnUint kvDim = DIFFUSION_ATTENTION_KV_HEADS * DIFFUSION_ATTENTION_HEAD_DIM;
    const NnUint kvGroup = DIFFUSION_ATTENTION_Q_HEADS / DIFFUSION_ATTENTION_KV_HEADS;
    for (NnUint qi = 0u; qi < queryLength; ++qi) {
        for (NnUint qh = 0u; qh < DIFFUSION_ATTENTION_Q_HEADS; ++qh) {
            const NnUint kvh = qh / kvGroup;
            const float *q = &query[qi * qDim + qh * DIFFUSION_ATTENTION_HEAD_DIM];
            float scores[DIFFUSION_ATTENTION_CACHE_CAPACITY + N_BATCHES];
            const float *values[DIFFUSION_ATTENTION_CACHE_CAPACITY + N_BATCHES];
            NnUint nScores = 0u;
            NnUint prefixStart = 0u;
            if (slidingWindow != 0u) {
                const NnUint end = mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL
                    ? (NnUint)positions[qi] + 1u
                    : prefixLength;
                prefixStart = end > slidingWindow ? end - slidingWindow : 0u;
            }
            NnUint prefixEnd = prefixLength;
            if (mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL)
                prefixEnd = std::min(prefixEnd, (NnUint)positions[qi] + 1u);
            for (NnUint absolute = prefixStart; absolute < prefixEnd; ++absolute) {
                const NnUint slot = slidingWindow == 0u
                    ? absolute
                    : absolute % DIFFUSION_ATTENTION_CACHE_CAPACITY;
                const NnUint offset = slot * kvDim + kvh * DIFFUSION_ATTENTION_HEAD_DIM;
                float score = 0.0f;
                for (NnUint d = 0u; d < DIFFUSION_ATTENTION_HEAD_DIM; ++d)
                    score += q[d] * keyCache[offset + d];
                scores[nScores] = score;
                values[nScores++] = &valueCache[offset];
            }
            for (NnUint ki = 0u; ki < queryLength; ++ki) {
                const NnUint keyPosition = (NnUint)positions[ki];
                if (mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL &&
                    keyPosition > (NnUint)positions[qi])
                    continue;
                if (mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL &&
                    slidingWindow != 0u && keyPosition < prefixStart)
                    continue;
                const NnUint offset = ki * kvDim + kvh * DIFFUSION_ATTENTION_HEAD_DIM;
                float score = 0.0f;
                for (NnUint d = 0u; d < DIFFUSION_ATTENTION_HEAD_DIM; ++d)
                    score += q[d] * currentKey[offset + d];
                scores[nScores] = score;
                values[nScores++] = &currentValue[offset];
            }
            float maximum = scores[0];
            for (NnUint i = 1u; i < nScores; ++i)
                maximum = std::max(maximum, scores[i]);
            float denominator = 0.0f;
            for (NnUint i = 0u; i < nScores; ++i) {
                scores[i] = expf(scores[i] - maximum);
                denominator += scores[i];
            }
            float *out = &output[qi * qDim + qh * DIFFUSION_ATTENTION_HEAD_DIM];
            for (NnUint d = 0u; d < DIFFUSION_ATTENTION_HEAD_DIM; ++d) {
                out[d] = 0.0f;
                for (NnUint i = 0u; i < nScores; ++i)
                    out[d] += scores[i] * values[i][d] / denominator;
            }
        }
    }
}

template <
    NnDiffusionAttentionMode mode,
    NnUint slidingWindow,
    NnUint prefixLengthValue,
    NnUint positionStart>
void testDiffusionAttention_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint queryPipeIndex = netBuilder->addPipe("query",
                size2D(F_32, N_BATCHES, DIFFUSION_ATTENTION_Q_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            const NnUint positionsPipeIndex = netBuilder->addPipe("positions", size2D(F_32, N_BATCHES, 1u));
            const NnUint prefixLengthPipeIndex = netBuilder->addPipe("prefix_length", size1D(F_32, 1u));
            const NnUint outputPipeIndex = netBuilder->addPipe("output",
                size2D(F_32, N_BATCHES, DIFFUSION_ATTENTION_Q_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            netBuilder->addPreSync(positionsPipeIndex);
            netBuilder->addPreSync(prefixLengthPipeIndex);

            const NnUint currentKeyBufferIndex = nodeBuilder->addBuffer("current_key",
                size2D(F_32, N_BATCHES, DIFFUSION_ATTENTION_KV_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            const NnUint currentValueBufferIndex = nodeBuilder->addBuffer("current_value",
                size2D(F_32, N_BATCHES, DIFFUSION_ATTENTION_KV_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            const NnUint keyCacheBufferIndex = nodeBuilder->addBuffer("key_cache",
                size1D(F_32, DIFFUSION_ATTENTION_CACHE_CAPACITY * DIFFUSION_ATTENTION_KV_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            const NnUint valueCacheBufferIndex = nodeBuilder->addBuffer("value_cache",
                size1D(F_32, DIFFUSION_ATTENTION_CACHE_CAPACITY * DIFFUSION_ATTENTION_KV_HEADS * DIFFUSION_ATTENTION_HEAD_DIM));
            segmentBuilder->addOp(OP_DIFFUSION_ATTENTION, "diffusion_attention", 0u,
                pointerBatchConfig(SRC_PIPE, queryPipeIndex),
                pointerBatchConfig(SRC_PIPE, outputPipeIndex), size0(),
                NnDiffusionAttentionOpConfig{
                    mode,
                    DIFFUSION_ATTENTION_Q_HEADS,
                    DIFFUSION_ATTENTION_KV_HEADS,
                    DIFFUSION_ATTENTION_Q_HEADS,
                    DIFFUSION_ATTENTION_KV_HEADS,
                    0u, 0u, DIFFUSION_ATTENTION_HEAD_DIM,
                    slidingWindow, DIFFUSION_ATTENTION_CACHE_CAPACITY, 1.0f,
                    positionsPipeIndex, prefixLengthPipeIndex,
                    currentKeyBufferIndex, currentValueBufferIndex,
                    keyCacheBufferIndex, valueCacheBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            const NnUint queryLength = 2u;
            execution->setBatchSize(queryLength);
            float *query = (float *)execution->pipes[0];
            const float queryValues[] = {
                1, 0, 0, 0,  0, 1, 0, 0,
                0, 0, 1, 0,  0, 0, 0, 1,
            };
            std::copy(queryValues, queryValues + 16u, query);
            float *positions = (float *)execution->pipes[1];
            positions[0] = (float)positionStart;
            positions[1] = (float)(positionStart + 1u);
            float *prefixLength = (float *)execution->pipes[2];
            prefixLength[0] = (float)prefixLengthValue;

            const float currentKey[] = {
                0.3f, 0.1f, 0.2f, 0.0f,
                0.0f, 0.4f, 0.1f, 0.2f,
            };
            const float currentValue[] = {
                3, 4, 5, 6,
                7, 8, 9, 10,
            };
            const float keyCache[] = {
                0.1f, 0.2f, 0.0f, 0.3f,
                0.2f, 0.0f, 0.4f, 0.1f,
                0.4f, 0.1f, 0.3f, 0.2f,
                0.0f, 0.3f, 0.2f, 0.4f,
            };
            const float valueCache[] = {
                1, 2, 3, 4,
                5, 6, 7, 8,
                9, 10, 11, 12,
                13, 14, 15, 16,
            };
            device->data.buffers[0].get()->write((const NnByte *)currentKey);
            device->data.buffers[1].get()->write((const NnByte *)currentValue);
            device->data.buffers[2].get()->write((const NnByte *)keyCache);
            device->data.buffers[3].get()->write((const NnByte *)valueCache);

            float expected[16];
            referenceDiffusionAttention(expected, query, currentKey, currentValue,
                keyCache, valueCache, positions, (NnUint)prefixLength[0], queryLength,
                mode, slidingWindow);
            executor->forward();

            const float *actual = (const float *)execution->pipes[3];
            for (NnUint i = 0u; i < 16u; ++i)
                assertFloat(i, actual[i], expected[i], 0.00001f);
            printOk(mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL
                ? (slidingWindow == 0u
                    ? "testDiffusionAttentionEncoderFull"
                    : "testDiffusionAttentionEncoderSliding")
                : (slidingWindow == 0u
                    ? "testDiffusionAttentionDecoderFull"
                    : "testDiffusionAttentionDecoderSliding"));
        });
}

template <NnUint dim>
void testShift_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, 1, N_BATCHES * dim));
            segmentBuilder->addOp(
                OP_SHIFT, "shift", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerRawConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnShiftOpCodeConfig{posPipeIndex, 0u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[1];
            float *yPipe = (float *)execution->pipes[2];

            float pos[N_BATCHES];
            for (NnUint b = 0; b < N_BATCHES; b++) {
                pos[b] = (float)b;
                for (NnUint i = 0; i < dim; i++)
                    xPipe[b * dim + i] = (float)(b * 100 + i);
            }

            device->data.pipes[0].get()->write((NnByte *)pos);

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < dim; i++) {
                    NnUint j = b * dim + i;
                    assertFloat(j, yPipe[j], (float)(b * 100 + i), 0.00001f);
                }
            }
            printOk("testShift_F32_F32");
        });
}

template <NnUint dim>
void testShiftRing_F32_F32() {
    const NnUint capacity = 4u;
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1u));
            const NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, dim));
            const NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, 1u, capacity * dim));
            segmentBuilder->addOp(
                OP_SHIFT, "shift_ring", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerRawConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnShiftOpCodeConfig{posPipeIndex, capacity});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            const NnUint batchSize = 3u;
            execution->setBatchSize(batchSize);
            float *xPipe = (float *)execution->pipes[1];
            float *yPipe = (float *)execution->pipes[2];
            for (NnUint i = 0u; i < batchSize * dim; ++i)
                xPipe[i] = (float)(i + 1u);
            std::fill(yPipe, yPipe + capacity * dim, -1.0f);

            const float positions[] = {3.0f, 4.0f, 5.0f};
            device->data.pipes[0].get()->write((const NnByte *)positions);
            device->data.pipes[2].get()->write((const NnByte *)yPipe);
            executor->forward();

            for (NnUint slot = 0u; slot < capacity; ++slot) {
                const NnUint sourceBatch = slot == 0u ? 1u : slot == 1u ? 2u : slot == 3u ? 0u : UINT32_MAX;
                for (NnUint i = 0u; i < dim; ++i) {
                    const float expected = sourceBatch == UINT32_MAX
                        ? -1.0f
                        : (float)(sourceBatch * dim + i + 1u);
                    assertFloat(slot * dim + i, yPipe[slot * dim + i], expected, 0.00001f);
                }
            }
            printOk("testShiftRing_F32_F32");
        });
}

template <NnUint dim, NnUint nZ>
void testCast_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_32, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(
                OP_CAST, "cast", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float *yPipe = (float *)execution->pipes[1];

            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                xPipe[i] = (float)(i + 1);

            // act
            executor->forward();

            // assert
            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                assertFloat(i, yPipe[i], (float)(i + 1), 0.00001f);
            printOk("testCast_F32_F32");
        });
}

void testCastBatchZ_F32_F32() {
    const NnUint dim = 8u;
    const NnUint nZ = 2u;
    execute(
        [](NnNetConfigBuilder *, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint xBufferIndex = nodeBuilder->addBuffer("X", size3D(F_32, nZ, N_BATCHES, dim));
            const NnUint yBufferIndex = nodeBuilder->addBuffer("Y", size3D(F_32, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(
                OP_CAST, "cast_batch_z", 0u,
                pointerBatchZConfig(SRC_BUFFER, xBufferIndex, 1u),
                pointerBatchZConfig(SRC_BUFFER, yBufferIndex, 0u),
                size0(), NnCastOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            const NnUint batchSize = 3u;
            execution->setBatchSize(batchSize);
            float x[nZ * N_BATCHES * dim];
            float y[nZ * N_BATCHES * dim];
            for (NnUint i = 0u; i < nZ * N_BATCHES * dim; ++i) {
                x[i] = (float)(i + 1u);
                y[i] = -1.0f;
            }
            device->data.buffers[0].get()->write((const NnByte *)x);
            device->data.buffers[1].get()->write((const NnByte *)y);

            executor->forward();
            device->data.buffers[1].get()->read((NnByte *)y);

            for (NnUint batch = 0u; batch < N_BATCHES; ++batch) {
                for (NnUint i = 0u; i < dim; ++i) {
                    const NnUint destination = batch * dim + i;
                    const NnUint source = (N_BATCHES + batch) * dim + i;
                    const float expected = batch < batchSize ? x[source] : -1.0f;
                    assertFloat(destination, y[destination], expected, 0.00001f);
                }
            }
            for (NnUint i = N_BATCHES * dim; i < nZ * N_BATCHES * dim; ++i)
                assertFloat(i, y[i], -1.0f, 0.00001f);
            printOk("testCastBatchZ_F32_F32");
        });
}

template <NnUint dim, NnUint nZ>
void testCast_F32_Q80() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_Q80, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(
                OP_CAST, "cast", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size0(),
                NnCastOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            NnBlockQ80 *yPipe = (NnBlockQ80 *)execution->pipes[1];

            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++)
                xPipe[i] = (float)(i + 1);

            // act
            executor->forward();

            float yF32[nZ * N_BATCHES * dim];
            dequantizeQ80toF32(yPipe, yF32, nZ * N_BATCHES * dim, 1, 0);

            for (NnUint i = 0; i < nZ * N_BATCHES * dim; i++) {
                const float expectedV = (float)(i + 1);
                const float change = (yF32[i] - expectedV) / expectedV;
                assertFloat(i, change, 0.0, 0.009f);
            }
            printOk("testCast_F32_Q80");
        });
}

template <NnRopeType ropeType, void (*assertOutput)(float *x0, float *x1)>
void testRope_F32_F32() {
    #define ROPE_DIM 2048
    #define ROPE_KV_DIM 512
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 32;
            const NnUint seqLen = 4096;
            const NnRopeSlice slice = sliceRope(ropeType, ROPE_DIM, ROPE_KV_DIM, 8, 1, seqLen, ROPE_DIM / nHeads, 500000.0f, 0);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, ROPE_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint ropeCacheBufferIndex = nodeBuilder->addBuffer("ropeCache", slice.cacheSize);
            NnUint isQ = 1;

            segmentBuilder->addOp(
                OP_ROPE, "rope_llama", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnRopeOpConfig{ropeType, isQ, posPipeIndex, ropeCacheBufferIndex, 32.0f, 1.0f, 4.0f, 8192, 0u, slice});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(2);

            float *xPipe = (float *)execution->pipes[0];
            float pos[N_BATCHES];
            pos[0] = (float)6;
            pos[1] = (float)31;

            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint i = 0; i < ROPE_DIM; i++)
                    xPipe[b * ROPE_DIM + i] = 1.0f;
            }

            device->data.pipes[1].get()->write((NnByte *)pos);

            // act
            executor->forward();

            // assert
            float *x0 = &xPipe[0 * ROPE_DIM];
            float *x1 = &xPipe[1 * ROPE_DIM];
            assertOutput(x0, x1);
        });
}

void assertRopeLlama_F32_F32(float *x0, float *x1) {
    const float t = 0.000001f;

    assertFloat(0, x0[0], 1.239586f, t);
    assertFloat(1, x0[1], 0.680755f, t);
    assertFloat(2, x0[2], 0.077202f, t);
    assertFloat(3, x0[3], -1.412105f, t);
    assertFloat(1988, x0[1988], -1.356766f, t);
    assertFloat(2022, x0[2022], 0.999923f, t);
    assertFloat(2023, x0[2023], 1.000077f, t);

    assertFloat(0, x1[0], 1.318780f, t);
    assertFloat(1, x1[1], 0.510705f, t);
    assertFloat(1078, x1[1078], 0.999985f, t);
    assertFloat(1078, x1[1079], 1.000015f, t);
}

void assertRopeFalcon_F32_F32(float *x0, float *x1) {
    const float t = 0.000001f;

    assertFloat(0, x0[0], 1.239586f, t);
    assertFloat(1, x0[1], 0.077202f, t);
    assertFloat(2, x0[2], -1.356766f, t);
    assertFloat(3, x0[3], -1.164938f, t);
    assertFloat(1988, x0[1988], -0.522115f, t);
    assertFloat(1988, x0[1989], 0.018772f, t);
    assertFloat(2022, x0[2022], 1.361834f, t);
    assertFloat(2023, x0[2023], 1.276253f, t);

    assertFloat(0, x1[0], 1.318780f, t);
    assertFloat(1, x1[1], -1.139289f, t);
    assertFloat(1, x1[2], -0.417384f, t);
    assertFloat(1, x1[3], -1.291486f, t);
    assertFloat(1078, x1[1078], 1.003737f, t);
    assertFloat(1078, x1[1079], 1.002481f, t);
}

void testRopeLlama_F32_F32() {
    testRope_F32_F32<NnRopeType::ROPE_LLAMA, assertRopeLlama_F32_F32>();
    printOk("testRopeLlama_F32_F32");
}

void testRopeFalcon_F32_F32() {
    testRope_F32_F32<NnRopeType::ROPE_FALCON, assertRopeFalcon_F32_F32>();
    printOk("testRopeFalcon_F32_F32");
}

void testRopeProportionalPartial_F32_F32() {
    #define ROPE_PROPORTIONAL_DIM 8
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnRopeSlice slice = sliceRope(
                ROPE_PROPORTIONAL, ROPE_PROPORTIONAL_DIM, ROPE_PROPORTIONAL_DIM,
                1u, 1u, 16u, ROPE_PROPORTIONAL_DIM, 10000.0f, 0u);
            const NnUint xPipeIndex = netBuilder->addPipe(
                "X", size2D(F_32, N_BATCHES, ROPE_PROPORTIONAL_DIM));
            const NnUint positionPipeIndex = netBuilder->addPipe(
                "POS", size2D(F_32, N_BATCHES, 1u));
            const NnUint ropeCacheBufferIndex = nodeBuilder->addBuffer("ropeCache", slice.cacheSize);
            segmentBuilder->addOp(OP_ROPE, "rope_proportional", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex), pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(), NnRopeOpConfig{
                    ROPE_PROPORTIONAL, 1u, positionPipeIndex, ropeCacheBufferIndex,
                    32.0f, 0.0f, 0.0f, 0u, 4u, slice});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            execution->setBatchSize(N_BATCHES);
            float *x = (float *)execution->pipes[0];
            float positions[N_BATCHES];
            for (NnUint batch = 0u; batch < N_BATCHES; ++batch) {
                positions[batch] = 2.0f;
                for (NnUint h = 0u; h < ROPE_PROPORTIONAL_DIM; ++h)
                    x[batch * ROPE_PROPORTIONAL_DIM + h] = (float)(h + 1u);
            }
            device->data.pipes[1].get()->write((NnByte *)positions);
            executor->forward();

            const float angles[] = {
                2.0f / 32.0f,
                2.0f * (1.0f / std::pow(10000.0f, 2.0f / 8.0f)) / 32.0f,
            };
            float expected[ROPE_PROPORTIONAL_DIM] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
            for (NnUint j = 0u; j < 2u; ++j) {
                const float c = std::cos(angles[j]);
                const float s = std::sin(angles[j]);
                expected[j] = (float)(j + 1u) * c - (float)(j + 5u) * s;
                expected[j + 4u] = (float)(j + 1u) * s + (float)(j + 5u) * c;
            }
            for (NnUint batch = 0u; batch < N_BATCHES; ++batch)
                for (NnUint h = 0u; h < ROPE_PROPORTIONAL_DIM; ++h)
                    assertFloat(batch * ROPE_PROPORTIONAL_DIM + h,
                        x[batch * ROPE_PROPORTIONAL_DIM + h], expected[h], 0.00001f);
            printOk("testRopeProportionalPartial_F32_F32");
        });
}

template <NnUint N, NnUint D>
void testMatmul_F32_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, N));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, N_BATCHES, D));
            NnUint nullBufferIndex = nodeBuilder->addBuffer("null", size1D(F_32, 1u));
            segmentBuilder->addOp(
                OP_MATMUL, "matmul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size2D(F_32, N, D),
                NnMatmulOpConfig{0u, 0u, nullBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0];
            float *yPipe = (float *)execution->pipes[1];

            float weight[N * D];
            for (NnUint i = 0; i < N_BATCHES * N; i++)
                xPipe[i] = i * 0.0001f;
            for (NnUint i = 0; i < N * D; i++)
                weight[i] = i * 0.000001f;
            executor->loadWeight("matmul", 0u, 0u, N * D * sizeof(float), (NnByte *)weight);

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (NnUint n = 0; n < N; n++)
                        sum += xPipe[b * N + n] * weight[d * N + n];

                    const NnUint p = b * D + d;
                    assertFloat(p, yPipe[p], sum, 0.0002f);
                }
            }
            printOk("testMatmul_F32_F32_F32");
        });
}

void testMatmul_F32_F32_F32_expert() {
    #define MATMUL_F32_N 4
    #define MATMUL_F32_D 1
    #define MATMUL_F32_E 4
    #define MATMUL_F32_A 2
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, MATMUL_F32_A, N_BATCHES, MATMUL_F32_N));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size3D(F_32, MATMUL_F32_A, N_BATCHES, MATMUL_F32_D));
            NnUint activeExpertIndexesIndex = nodeBuilder->addBuffer("indexes", size2D(F_32, N_BATCHES, MATMUL_F32_A));
            segmentBuilder->addOp(
                OP_MATMUL, "matmul", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size3D(F_32, MATMUL_F32_E, MATMUL_F32_N, MATMUL_F32_D),
                NnMatmulOpConfig{MATMUL_F32_E, MATMUL_F32_A, activeExpertIndexesIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            float *xPipe = (float *)execution->pipes[0]; // (A, N_BATCHES, N)
            float *yPipe = (float *)execution->pipes[1]; // (A, N_BATCHES, D)

            constexpr NnUint wSize = MATMUL_F32_N * MATMUL_F32_D;
            constexpr NnUint wSizeBytes = wSize * sizeof(float);
            float weight[wSize];
            float indexes[N_BATCHES * MATMUL_F32_A];

            for (NnUint e = 0u; e < MATMUL_F32_E; e++) {
                for (NnUint i = 0u; i < wSize; i++)
                    weight[i] = 0.1f * (float)(e + 1);
                executor->loadWeight("matmul", 0u, wSizeBytes * e, wSizeBytes, (NnByte *)weight);
            }

            for (NnUint i = 0u; i < MATMUL_F32_A * N_BATCHES; i++)
                indexes[i] = (float)(i % MATMUL_F32_E); // 0, 1, 2, 3, 0, 1, 2, 3, ...
            for (NnUint i = 0; i < MATMUL_F32_A * N_BATCHES * MATMUL_F32_N; i++)
                xPipe[i] = (float)(i / MATMUL_F32_N + 1); // 1.0, 1.0, ... 2.0, 2.0, ...

            device->data.buffers[0].get()->write((NnByte *)indexes);

            executor->forward();

            float t = 0.0001f;
            assertFloat(0, yPipe[0], 0.1f /* index=0, e=0 */ * (4 * 1.0f), t);
            assertFloat(1, yPipe[1], 0.3f /* index=2, e=2 */ * (4 * 2.0f), t);
            assertFloat(2, yPipe[2], 0.1f /* index=4, e=0 */ * (4 * 3.0f), t);
            assertFloat(3, yPipe[3], 0.3f /* index=6, e=2 */ * (4 * 4.0f), t);

            assertFloat(4, yPipe[4], 0.2f /* index=1, e=1 */ * (4 * 5.0f), t);
            assertFloat(5, yPipe[5], 0.4f /* index=3, e=3 */ * (4 * 6.0f), t);
            assertFloat(6, yPipe[6], 0.2f /* index=5, e=1 */ * (4 * 7.0f), t);
            assertFloat(7, yPipe[7], 0.4f /* index=7, e=3 */ * (4 * 8.0f), t);

            printOk("testMatmul_F32_F32_F32_expert");
        });
}

template <NnUint N, NnUint D>
void testMatmul_Q80_Q40_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_Q80, N_BATCHES, N));
            NnUint yPipeIndex = netBuilder->addPipe("Y", size2D(F_32, N_BATCHES, D));
            NnUint nullBufferIndex = nodeBuilder->addBuffer("null", size1D(F_32, 1u));
            segmentBuilder->addOp(
                OP_MATMUL, "matmul", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, yPipeIndex),
                size2D(F_Q40, N, D),
                NnMatmulOpConfig{0u, 0u, nullBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);
            NnBlockQ80 *xPipe = (NnBlockQ80 *)execution->pipes[0];
            float *yPipe = (float *)execution->pipes[1];

            constexpr NnUint xSize = N_BATCHES * N;
            constexpr NnUint weightSize = N * D;
            constexpr NnUint weightBlocks = weightSize / Q40_BLOCK_SIZE;

            std::unique_ptr<float[]> x(new float[xSize]);
            std::unique_ptr<float[]> weight(new float[weightSize]);
            std::unique_ptr<NnBlockQ40[]> weightQ40(new NnBlockQ40[weightBlocks]);

            for (NnUint i = 0; i < xSize; i++)
                x[i] = 0.1f + (i / (float)N - 0.5f) * 0.0005f;
            for (NnUint i = 0; i < weightSize; i++)
                weight[i] = 0.1f + (i / (float)D - 0.5f) * 0.0005f;

            quantizeF32toQ80(x.get(), xPipe, xSize, 1, 0);
            quantizeF32toQ40(weight.get(), weightQ40.get(), weightSize, 1, 0);

            executor->loadWeight("matmul", 0u, 0u, weightBlocks * sizeof(NnBlockQ40), (NnByte *)weightQ40.get());

            // act
            executor->forward();

            // assert
            for (NnUint b = 0; b < N_BATCHES; b++) {
                for (NnUint d = 0; d < D; d++) {
                    float sum = 0.0f;
                    for (NnUint n = 0; n < N; n++)
                        sum += x[b * N + n] * weight[d * N + n];
                    const NnUint p = b * D + d;

                    const float err = sum == 0.0 ? (yPipe[p] - sum) : (yPipe[p] - sum) / sum;
                    // printf("[%d] %f %f (%f)\n", b, yPipe[p], sum, err);
                    const float tolerance = D % 8u == 0u ? 0.009f : 0.010f;
                    assertFloat(p, err, 0.0f, tolerance);
                }
            }
            printOk("testMatmul_Q80_Q40_F32");
        });
}

void testMultiheadAtt_F32_F32() {
    #define MULTIHEAD_ATT_DIM 128
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint nHeads = 32;
            const NnUint nKvHeads = 8;
            const NnUint headDim = MULTIHEAD_ATT_DIM / nHeads;
            const NnUint seqLen = 4096;
            const NnUint qSliceD0 = 2048;
            const NnUint kvDim0 = 512;
            const NnKvCacheSlice kvCacheSlice = sliceKvCache(kvDim0, seqLen, 1);
            const NnMultiHeadAttSlice multiHeadAttSlice = sliceMultiHeadAtt(nHeads, seqLen, 1, N_BATCHES);

            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MULTIHEAD_ATT_DIM));
            NnUint posPipeIndex = netBuilder->addPipe("POS", size2D(F_32, N_BATCHES, 1));
            NnUint qBufferIndex = nodeBuilder->addBuffer("POS", size2D(F_32, N_BATCHES, qSliceD0));
            NnUint kCacheBufferIndex = nodeBuilder->addBuffer("kCache", kvCacheSlice.keySize);
            NnUint vCacheBufferIndex = nodeBuilder->addBuffer("vCache", kvCacheSlice.valueSize);
            NnUint attCacheBufferIndex = nodeBuilder->addBuffer("vCache", multiHeadAttSlice.attSize);

            segmentBuilder->addOp(
                OP_MULTIHEAD_ATT, "multihead_att", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnMultiHeadAttOpConfig{nHeads, nHeads, nKvHeads, headDim, seqLen, qSliceD0, kvDim0,
                    posPipeIndex, qBufferIndex, kCacheBufferIndex, vCacheBufferIndex, attCacheBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // TODO: for now this is a smoke test
            execution->setBatchSize(N_BATCHES);
            executor->forward();
            printOk("testMultiheadAtt_F32_F32");
        });
}

template <NnUint dim, NnUint nZ>
void testSoftmax_F32_F32() {
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, nZ, N_BATCHES, dim));
            segmentBuilder->addOp(OP_SOFTMAX, "softmax", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnSoftmaxOpCodeConfig{});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];

            for (NnUint z = 0u; z < nZ; z++) {
                float *xPipeZ = &xPipe[z * N_BATCHES * dim];
                for (NnUint b = 0; b < N_BATCHES; b++) {
                    const NnUint offset = b * dim;
                    for (NnUint i = 0u; i < dim; i++)
                        xPipeZ[offset + i] = i / (float)dim + (float)b;
                }
            }

            // act
            executor->forward();

            // assert
            float t = 0.00001f;
            for (NnUint z = 0u; z < nZ; z++) {
                float *xPipeZ = &xPipe[z * N_BATCHES * dim];
                for (NnUint b = 0; b < N_BATCHES; b++) {
                    const NnUint offset = b * dim;
                    float max = ((dim - 1) / (float)dim) + (float)b;
                    for (NnUint i = 0u; i < dim; i++) {
                        const float v = i / (float)dim + (float)b;
                        const float expectedOutput = expf(v - max);
                        assertFloat(offset + i, xPipeZ[offset + i], expectedOutput, t);
                    }
                }
            }

            printOk("testSoftmax_F32_F32");
        });
}

void testScale_F32_F32() {
    #define SCALE_F32_N_Z 4
    #define SCALE_F32_DIM 64
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size3D(F_32, SCALE_F32_N_Z, N_BATCHES, SCALE_F32_DIM));
            NnUint scaleBufferIndex = nodeBuilder->addBuffer("scale", size3D(F_32, SCALE_F32_N_Z, SCALE_F32_DIM, 1u));
            segmentBuilder->addOp(OP_SCALE, "scale", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                size0(),
                NnScaleOpCodeConfig{scaleBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float scale[SCALE_F32_N_Z * SCALE_F32_DIM];

            for (NnUint z = 0u; z < SCALE_F32_N_Z; z++) {
                const NnUint zOffset = z * N_BATCHES * SCALE_F32_DIM;
                for (NnUint y = 0u; y < N_BATCHES; y++) {
                    scale[z * N_BATCHES + y] = 0.5f * (float)(z * 100 + y);
                    for (NnUint i = 0; i < SCALE_F32_DIM; i++)
                        xPipe[zOffset + y * SCALE_F32_DIM + i] = (float)(z * 10000 + y * 100 + i);
                }
            }

            device->data.buffers[0].get()->write((NnByte *)scale);

            // act
            executor->forward();

            // assert
            for (NnUint z = 0u; z < SCALE_F32_N_Z; z++) {
                const NnUint zOffset = z * N_BATCHES * SCALE_F32_DIM;
                for (NnUint y = 0u; y < N_BATCHES; y++) {
                    for (NnUint i = 0; i < SCALE_F32_DIM; i++) {
                        const NnUint p = zOffset + y * SCALE_F32_DIM + i;
                        const float v = xPipe[p];
                        const float expectedOutput = (float)(z * 10000 + y * 100 + i) * 0.5f * (float)(z * 100 + y);
                        assertFloat(p, v, expectedOutput, 0.00001f);
                    }
                }
            }

            printOk("testScale_F32_F32");
        });
}

void testScaleScalar_F32_F32() {
    #define SCALE_SCALAR_DIM 32
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint inputPipeIndex = netBuilder->addPipe(
                "input", size2D(F_32, N_BATCHES, SCALE_SCALAR_DIM));
            const NnUint outputPipeIndex = netBuilder->addPipe(
                "output", size2D(F_32, N_BATCHES, SCALE_SCALAR_DIM));
            segmentBuilder->addOp(OP_SCALE, "scale_scalar", 0u,
                pointerBatchConfig(SRC_PIPE, inputPipeIndex), pointerBatchConfig(SRC_PIPE, outputPipeIndex),
                size0(), NnScaleOpCodeConfig{0u, 0.25f, 1u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *) {
            execution->setBatchSize(N_BATCHES);
            float *input = (float *)execution->pipes[0];
            for (NnUint i = 0u; i < N_BATCHES * SCALE_SCALAR_DIM; ++i)
                input[i] = (float)((int)i - 37) * 0.125f;
            executor->forward();
            const float *output = (const float *)execution->pipes[1];
            for (NnUint i = 0u; i < N_BATCHES * SCALE_SCALAR_DIM; ++i)
                assertFloat(i, output[i], input[i] * 0.25f, 0.00001f);
            printOk("testScaleScalar_F32_F32");
        });
}

void testMoeGate_F32_F32() {
    #define MOE_GATE_F32_DIM 8
    #define MOE_GATE_F32_K 4
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            NnUint xPipeIndex = netBuilder->addPipe("X", size2D(F_32, N_BATCHES, MOE_GATE_F32_DIM));
            NnUint gPipeIndex = netBuilder->addPipe("g", size3D(F_32, MOE_GATE_F32_K, N_BATCHES, 1u));
            NnUint indexesBufferIndex = nodeBuilder->addBuffer("i", size2D(F_32, N_BATCHES, MOE_GATE_F32_K));
            segmentBuilder->addOp(OP_MOE_GATE, "moe_gate", 0,
                pointerBatchConfig(SRC_PIPE, xPipeIndex),
                pointerBatchConfig(SRC_PIPE, gPipeIndex),
                size0(),
                NnMoeGateOpCodeConfig{MOE_GATE_F32_K, 0u, indexesBufferIndex});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            // arrange
            execution->setBatchSize(N_BATCHES);

            float *xPipe = (float *)execution->pipes[0];
            float *gPipe = (float *)execution->pipes[1];
            xPipe[0] = 3.0f;
            xPipe[1] = 1.0f;
            xPipe[2] = 6.0f;
            xPipe[3] = 5.0f;
            xPipe[4] = 8.0f;
            xPipe[5] = 4.0f;
            xPipe[6] = 2.0f;
            xPipe[7] = 7.0f;

            // act
            executor->forward();

            float pos[N_BATCHES * MOE_GATE_F32_K];
            device->data.buffers[0].get()->read((NnByte *)pos);

            // assert
            const float t = 0.00001f;
            assertFloat(0, gPipe[0 * N_BATCHES], 8.0f, t);
            assertFloat(1, gPipe[1 * N_BATCHES], 7.0f, t);
            assertFloat(2, gPipe[2 * N_BATCHES], 6.0f, t);
            assertFloat(3, gPipe[3 * N_BATCHES], 5.0f, t);

            assertFloat(100, pos[0], 4.0f, t);
            assertFloat(101, pos[1], 7.0f, t);
            assertFloat(102, pos[2], 2.0f, t);
            assertFloat(103, pos[3], 3.0f, t);

            printOk("testMoeGate_F32_F32");
        });
}

void testMoeGateWeightScale_F32_F32() {
    #define MOE_GATE_WEIGHT_DIM 8
    #define MOE_GATE_WEIGHT_K 4
    execute(
        [](NnNetConfigBuilder *netBuilder, NnNodeConfigBuilder *nodeBuilder, NnSegmentConfigBuilder *segmentBuilder) {
            const NnUint xPipeIndex = netBuilder->addPipe(
                "X", size2D(F_32, N_BATCHES, MOE_GATE_WEIGHT_DIM));
            const NnUint gPipeIndex = netBuilder->addPipe(
                "g", size3D(F_32, MOE_GATE_WEIGHT_K, N_BATCHES, 1u));
            const NnUint indexesBufferIndex = nodeBuilder->addBuffer(
                "i", size2D(F_32, N_BATCHES, MOE_GATE_WEIGHT_K));
            segmentBuilder->addOp(OP_MOE_GATE, "moe_gate_weight", 0u,
                pointerBatchConfig(SRC_PIPE, xPipeIndex), pointerBatchConfig(SRC_PIPE, gPipeIndex),
                size1D(F_32, MOE_GATE_WEIGHT_DIM),
                NnMoeGateOpCodeConfig{
                    MOE_GATE_WEIGHT_K, 1u, indexesBufferIndex, 0u, 1u, 1u});
        },
        [](NnExecutor *executor, NnNetExecution *execution, NnVulkanDevice *device) {
            execution->setBatchSize(N_BATCHES);
            const float scores[MOE_GATE_WEIGHT_DIM] = {3.0f, 1.0f, 6.0f, 5.0f, 8.0f, 4.0f, 2.0f, 7.0f};
            const float scales[MOE_GATE_WEIGHT_DIM] = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f};
            float *x = (float *)execution->pipes[0];
            for (NnUint batch = 0u; batch < N_BATCHES; ++batch)
                std::copy(scores, scores + MOE_GATE_WEIGHT_DIM,
                    x + batch * MOE_GATE_WEIGHT_DIM);
            executor->loadWeight("moe_gate_weight", 0u, 0u, sizeof(scales), (NnByte *)scales);
            executor->forward();

            const NnUint expectedIndexes[MOE_GATE_WEIGHT_K] = {4u, 7u, 2u, 3u};
            const float expectedScores[MOE_GATE_WEIGHT_K] = {8.0f, 7.0f, 6.0f, 5.0f};
            const float *g = (const float *)execution->pipes[1];
            float indexes[N_BATCHES * MOE_GATE_WEIGHT_K];
            device->data.buffers[0].get()->read((NnByte *)indexes);
            for (NnUint selected = 0u; selected < MOE_GATE_WEIGHT_K; ++selected)
                for (NnUint batch = 0u; batch < N_BATCHES; ++batch) {
                    const NnUint index = selected * N_BATCHES + batch;
                    assertFloat(index, g[index],
                        expectedScores[selected] / 26.0f * scales[expectedIndexes[selected]], 0.00001f);
                    assertFloat(1000u + batch * MOE_GATE_WEIGHT_K + selected,
                        indexes[batch * MOE_GATE_WEIGHT_K + selected],
                        (float)expectedIndexes[selected], 0.00001f);
                }
            printOk("testMoeGateWeightScale_F32_F32");
        });
}

int main() {
    initQuants();

    testRmsNorm_F32_F32_F32<4>();
    testRmsNorm_F32_F32_F32<1024>();
    testRmsNorm_F32_F32_F32<3196>();

    testRmsNormScaleFree_F32_F32<4>();
    testRmsNormScaleFree_F32_F32<32>();

    testSilu_F32_F32<4>();
    testSilu_F32_F32<32>();
    testSilu_F32_F32<104>();

    testTanh_F32_F32<4>();
    testTanh_F32_F32<32>();

    testGelu_F32_F32<4>();
    testGelu_F32_F32<32>();

    testMul_F32_F32<32, 1>();
    testMul_F32_F32<48, 4>();
    testMulWeight_F32_F32();

    testMergeAdd_F32_F32();

    testMergeSum_F32_F32();
    testGroupedMergeSum_F32<false>();
    testGroupedMergeSum_F32<true>();

    testMergeAdd_Q80_F32<2, 64>();
    testMergeAdd_Q80_F32<4, 128>();
    testMergeAdd_Q80_F32<4, 160>();

    testEmbedding_F32_F32();
    testShardedEmbedding_F32<false>();
    testShardedEmbedding_F32<true>();

    testReduceMax_F32_F32<false>();
    testReduceMax_F32_F32<true>();
    testEmbeddingMixRescale_F32_F32();
    testVocabOps_F32_F32();

    testTiedEmbedding_F32<false>();
    testTiedEmbedding_F32<true, 32u, 32u>();
    testTiedEmbedding_F32<true, 256u, 96u, 128u, 9u, 9u>();
    testTiedEmbedding_F32<true, 128u, 32u, 65u, 3u, 9u>();
    testTiedEmbedding_F32<true, 256u, 2816u, 128u, 9u, 9u>();
    testTiedEmbedding_F32<true, 256u, 32u, 128u, 256u, 256u>();

    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_ENCODER_CAUSAL, 0u, 2u, 2u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL, 0u, 0u, 0u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL, 0u, 2u, 2u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_ENCODER_CAUSAL, 2u, 4u, 4u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL, 2u, 4u, 4u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL, 4u, 4u, 4u>();
    testDiffusionAttention_F32_F32<DIFFUSION_ATTENTION_ENCODER_CAUSAL, 4u, 10u, 10u>();

    testShift_F32_F32<32>();
    testShift_F32_F32<9>();
    testShiftRing_F32_F32<32>();

    testCast_F32_F32<128, 1>();
    testCast_F32_F32<32, 2>();
    testCast_F32_F32<8, 4>();
    testCastBatchZ_F32_F32();

    testCast_F32_Q80<256, 1>();
    testCast_F32_Q80<64, 4>();

    testRepeatZ_F32_F32();

    testRopeLlama_F32_F32();
    testRopeFalcon_F32_F32();
    testRopeProportionalPartial_F32_F32();

    testMatmul_F32_F32_F32<64, 96>();
    testMatmul_F32_F32_F32<3191, 109>();

    testMatmul_F32_F32_F32_expert();

    testMatmul_Q80_Q40_F32<14336, 4096>();
    testMatmul_Q80_Q40_F32<4096, 14336>();
    testMatmul_Q80_Q40_F32<4096, 4096>();
    testMatmul_Q80_Q40_F32<64, 48>();
    testMatmul_Q80_Q40_F32<64, 64>();
    testMatmul_Q80_Q40_F32<192, 16>();
    testMatmul_Q80_Q40_F32<32, 10>();

    testMultiheadAtt_F32_F32();

    testSoftmax_F32_F32<256, 1>();
    testSoftmax_F32_F32<512, 4>();

    testScale_F32_F32();
    testScaleScalar_F32_F32();

    testMoeGate_F32_F32();
    testMoeGateWeightScale_F32_F32();
    return 0;
}
