#include "nn-cpu-ops.cpp"
#include <vector>
#include <thread>

// framework

void printPassed(const char *name) {
    printf("✅ %24s passed\n", name);
    fflush(stdout);
}

void rand(float *o, const NnUint n, const NnUint seed) {
    srand(seed + 123456);
    for (NnUint i = 0; i < n; i++) {
        float v = (float)(rand() / RAND_MAX);
        o[i] = v * 2.0f - 1.0f;
    }
}

void compare_F32(const char *name, const float *a, const float *b, const NnUint n, const float epsilon) {
    for (NnUint i = 0; i < n; i++) {
        float error = fabs(a[i] - b[i]);
        if (error > epsilon) {
            printf("❌ %s failed\n", name);
            for (NnUint j = i; j < i + 16 && j < n; j++)
                printf("   [%3d] %f != %f\n", j, a[j], b[j]);
            exit(1);
        }
    }
    printPassed(name);
}

// tests

void testSplitThreads() {
    // <0; 32> across 3 threads
    {
        SPLIT_THREADS(a0Start, a0End, 32, 3, 0); // thread 0
        assert(a0Start == 0);
        assert(a0End == 11);
    }
    {
        SPLIT_THREADS(a1Start, a1End, 32, 3, 1); // thread 1
        assert(a1Start == 11);
        assert(a1End == 22);
    }
    {
        SPLIT_THREADS(a2Start, a2End, 32, 3, 2); // thread 2
        assert(a2Start == 22);
        assert(a2End == 32);
    }

    // <0; 4> across 8 threads
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 0); // thread 0
        assert(b0Start == 0);
        assert(b0End == 1);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 3); // thread 3
        assert(b0Start == 3);
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 4); // thread 4
        assert(b0Start == 4); 
        assert(b0End == 4);
    }
    {
        SPLIT_THREADS(b0Start, b0End, 4, 8, 7); // thread 7
        assert(b0Start == 4);
        assert(b0End == 4);
    }

    printPassed("splitThreads");
}

void testConvertF32toF16() {
    float x[] = {0.0f, 0.25f, 0.3456f, 1.0f};
    for (NnUint i = 0; i < sizeof(x) / sizeof(float); i++) {
        NnFp16 f16 = CONVERT_F32_TO_F16(x[i]);
        float f32 = CONVERT_F16_TO_F32(f16);
        compare_F32("convertF32toF16", &x[i], &f32, 1, 0.0005);
    }
}

// quantization
void testQuantization(const NnUint m) {
    std::vector<float> a(m * Q40_BLOCK_SIZE);
    std::vector<float> aTemp(m * Q40_BLOCK_SIZE);
    std::vector<NnBlockQ40> aQ40(m);
    std::vector<NnBlockQ80> aQ80(m);

    rand(a.data(), m * Q40_BLOCK_SIZE, m);

    quantizeF32toQ40(a.data(), aQ40.data(), m * Q40_BLOCK_SIZE, 1, 0);
    dequantizeQ40toF32(aQ40.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q40", a.data(), aTemp.data(), m * Q40_BLOCK_SIZE, 0.13);

    quantizeF32toQ80(a.data(), aQ80.data(), m * Q80_BLOCK_SIZE, 1, 0);
    dequantizeQ80toF32(aQ80.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 1, 0);

    compare_F32("testQuantization_Q80", a.data(), aTemp.data(), m * Q80_BLOCK_SIZE, 0.01);
}

// invRms
void testInvRms() {
    const float epsilon = 0.00001;

    std::vector<float> x(8);
    x[0] = 0.1f;
    x[1] = 0.3f;
    x[2] = 0.2f;
    x[3] = 0.4f;
    x[4] = 0.6f;
    x[5] = 0.5f;
    x[6] = 0.0f;
    x[7] = 0.8f;

    const float y0 = invRms_F32(x.data(), 8, epsilon);
    float ev0 = 1.0f / 0.4402f;
    compare_F32("rms_F32", &y0, &ev0, 1, 0.001f);
}

// rmsNorm
void testRmsNorm(const NnUint m) {
    std::vector<float> x(m);
    std::vector<NnBlockQ80> xQ80(m / Q80_BLOCK_SIZE);
    std::vector<float> w(m);
    std::vector<float> y(m);
    std::vector<float> yTemp(m);

    rand(x.data(), m, m);
    rand(w.data(), m, m * m);
    quantizeF32toQ80(x.data(), xQ80.data(), m, 1, 0);
    const float rms = invRms_F32(x.data(), m, 1e-5f);

    rmsNorm_F32(y.data(), x.data(), rms, w.data(), m, 1, 0);
    rmsNorm_Q80_F32_F32(yTemp.data(), xQ80.data(), rms, w.data(), m, 1, 0);

    compare_F32("rmsNorm_Q80_F32_F32", y.data(), yTemp.data(), m, 0.01);
}

void testScaleFreeRmsNorm() {
    float input[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[4];
    rmsNorm_F32(output, input, 0.25f, nullptr, 4u, 1u, 0u);

    float expected[] = {0.25f, 0.5f, 0.75f, 1.0f};
    compare_F32("scaleFreeRmsNorm_F32", output, expected, 4u, 0.000001f);
}

// a *= b
void testMul(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

    std::vector<float> a0(n);
    std::vector<float> b0(n);

    std::vector<float> aQ(n);
    std::vector<NnBlockQ80> b1(n / Q80_BLOCK_SIZE);

    rand(a0.data(), n, m);
    rand(aQ.data(), n, m);
    rand(b0.data(), n, m);
    quantizeF32toQ80(b0.data(), b1.data(), n, 1, 0);

    mul_F32(a0.data(), a0.data(), b0.data(), n, 1, 0);
    mul_Q80_F32(aQ.data(), aQ.data(), b1.data(), n, 1, 0);

    compare_F32("mul_Q80_F32", a0.data(), aQ.data(), n, 0.005);
}

// y += x
void testAdd(const NnUint m) {
    const NnUint n = Q80_BLOCK_SIZE * m;

    std::vector<float> y(n);
    std::vector<float> yTemp(n);
    std::vector<float> x(n);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);

    rand(y.data(), n, m);
    rand(yTemp.data(), n, m);
    rand(x.data(), n, m);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    add_F32(y.data(), x.data(), n, 1, 0);
    add_Q80_F32(yTemp.data(), xQ80.data(), n, 1, 0);

    compare_F32("add_Q80_F32", y.data(), yTemp.data(), n, 0.01);
}

void testMergeSum() {
    float inp[] = {
        /* [z0, y0] */ 0.1f, 0.2f,
        /* [z0, y1] */ 0.3f, 0.4f,
        /* [z1, y0] */ 0.5f, 0.6f,
        /* [z1, y1] */ 0.7f, 0.8f,
    };
    float out[4];

    float *i[4] = {
        &inp[0],
        &inp[2],
        &inp[4],
        &inp[6],
    };
    float *o[2] = {
        &out[0],
        &out[2]
    };

    mergeSum_F32(o, i, 2u, 2u, 2u, 2u, 1u, 0u);

    const float expectedOutput[4] = {
        0.6f,
        0.8f,
        1.0f,
        1.2f,
    };
    compare_F32("mergeSum_F32", out, expectedOutput, 4u, 0.00000001f);
}

void testSoftmax() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    softmax_F32(y.data(), 8);

    float expectedOutput[8] = {
        0.077399f,
        0.087780f,
        0.099500f,
        0.112761f,
        0.127778f,
        0.144793f,
        0.164072f,
        0.185917f
    };
    compare_F32("softmax_F32", y.data(), expectedOutput, 8, 0.001);
}

void testSilu() {
    std::vector<float> y(8);
    for (NnUint i = 0; i < 8; i++)
        y[i] = i / 8.0f;

    silu_F32(y.data(), 8, 1, 0);

    float expectedOutput[8] = {
        0.000000f,
        0.066401f,
        0.140544f,
        0.222250f,
        0.311233f,
        0.407116f,
        0.509461f,
        0.617802f
    };
    compare_F32("silu_F32", y.data(), expectedOutput, 8, 0.001);
}

// matmul
void testMatmul_F32_Q40_F32(const NnUint m = 2) {
    const NnUint n = Q80_BLOCK_SIZE * m;
    const NnUint d = Q80_BLOCK_SIZE * m;

    std::vector<float> x(n);
    std::vector<float> w(n * d);
    std::vector<float> o(d);
    std::vector<float> oTemp(d);
    std::vector<NnBlockQ80> xQ80(n / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ40> wQ40((n * d) / Q40_BLOCK_SIZE);

    rand(x.data(), n, m);
    rand(w.data(), n * d, m);
    quantizeF32toQ40(w.data(), wQ40.data(), n * d, 1, 0);
    quantizeF32toQ80(x.data(), xQ80.data(), n, 1, 0);

    matmul_F32_F32_F32(o.data(), x.data(), w.data(), n, d, 1, 0);

    matmul_Q80_Q40_F32(oTemp.data(), xQ80.data(), wQ40.data(), n, d, 1, 0);
    compare_F32("matmul_Q80_Q40_F32", o.data(), oTemp.data(), d, 4.0f);
}

void testLlamafileSgemm() {
    const NnUint batchSize = 8;
    const NnUint n = 256;
    const NnUint d = 128;

    std::vector<float> x(n * batchSize);
    std::vector<NnBlockQ80> xQ((n * batchSize) / Q80_BLOCK_SIZE);
    std::vector<float> w(n * d);
    std::vector<NnBlockQ40> wQ((n * d) / Q40_BLOCK_SIZE);
    std::vector<float> o(d * batchSize);
    std::vector<float> oTemp(d * batchSize);

    rand(x.data(), n * batchSize, 12345);
    rand(w.data(), n * d, 23456);

    quantizeF32toQ80(x.data(), xQ.data(), n * batchSize, 1, 0);
    quantizeF32toQ40(w.data(), wQ.data(), n * d, 1, 0);

    // f32

    for (NnUint i = 0; i < batchSize; i++) {
        matmul_F32_F32_F32(o.data() + i * d, x.data() + i * n, w.data(), n, d, 1, 0);
    }

    assert(llamafile_sgemm(
        d, batchSize, n,
        w.data(), n,
        x.data(), n,
        oTemp.data(), d,
        0, 1, 0,
        F_32, F_32, F_32
    ));

    compare_F32("llamafileSgemm_F32", o.data(), oTemp.data(), d * batchSize, 0.01f);

#if __ARM_FEATURE_DOTPROD
    // q40ᵀ * q80

    assert(llamafile_sgemm(
        d, batchSize, n / Q80_BLOCK_SIZE,
        wQ.data(), n / Q80_BLOCK_SIZE,
        xQ.data(), n / Q80_BLOCK_SIZE,
        oTemp.data(), d,
        0, 1, 0,
        F_Q40, F_Q80, F_32
    ));

    compare_F32("llamafileSgemm_Q80_Q40", o.data(), oTemp.data(), d * batchSize, 1.5f);
#endif
}

void testScale() {
    float i[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float o[4];
    scale_F32(i, o, 0.5f, 4u, 1u, 0u);
    float expectedOutput[] = {0.5f, 1.0f, 1.5f, 2.0f};
    compare_F32("scale_F32", o, expectedOutput, 4u, 0.00001f);
}

void testTanh() {
    float output[] = {-3.0f, -1.0f, 0.0f, 1.0f, 3.0f};
    float expected[5];
    for (NnUint i = 0u; i < 5u; i++)
        expected[i] = tanhf(output[i]);

    tanh_F32(output, 5u, 1u, 0u);
    compare_F32("tanh_F32", output, expected, 5u, 0.000001f);
}

void testShiftSlots() {
    assert(resolveShiftSlot(3.0f, 0u, 8u) == 3u);
    assert(resolveShiftSlot(11.0f, 8u, 8u) == 3u);

    float unique[] = {6.0f, 7.0f, 8.0f, 9.0f};
    validateShiftIndexes(unique, 4u, 8u, 8u);

    bool rejectedFraction = false;
    try {
        resolveShiftSlot(1.5f, 0u, 8u);
    } catch (const std::runtime_error &) {
        rejectedFraction = true;
    }
    assert(rejectedFraction);

    bool rejectedBounds = false;
    try {
        resolveShiftSlot(8.0f, 0u, 8u);
    } catch (const std::runtime_error &) {
        rejectedBounds = true;
    }
    assert(rejectedBounds);

    float duplicate[] = {1.0f, 9.0f};
    bool rejectedDuplicate = false;
    try {
        validateShiftIndexes(duplicate, 2u, 8u, 8u);
    } catch (const std::runtime_error &) {
        rejectedDuplicate = true;
    }
    assert(rejectedDuplicate);

    const NnUint width = 2u;
    const NnUint capacity = 4u;
    float input[] = {
        30.0f, 31.0f,
        40.0f, 41.0f,
        50.0f, 51.0f,
    };
    float output[] = {
        -1.0f, -1.0f,
        -2.0f, -2.0f,
        -3.0f, -3.0f,
        -4.0f, -4.0f,
    };
    float positions[] = {3.0f, 4.0f, 5.0f};
    NnByte *inputs[] = {
        (NnByte *)&input[0],
        (NnByte *)&input[width],
        (NnByte *)&input[width * 2u],
    };
    NnByte *outputs[] = {(NnByte *)output};
    NnByte *pipes[] = {(NnByte *)positions};
    NnShiftOpCodeConfig config = {0u, capacity};
    NnCpuOpContext context = {};
    context.nBatches = 3u;
    context.pipes = pipes;
    context.opConfig = &config;
    context.input = inputs;
    context.inputSize = size2D(F_32, 3u, width);
    context.hasInputContinuousMemory = true;
    context.output = outputs;
    context.outputSize = size2D(F_32, 1u, capacity * width);
    context.hasOutputContinuousMemory = true;

    initShiftForward(&context);
    shiftForward_F32_F32(1u, 0u, 3u, &context);
    const float expected[] = {
        40.0f, 41.0f,
        50.0f, 51.0f,
        -3.0f, -3.0f,
        30.0f, 31.0f,
    };
    compare_F32("shiftRingWrap", output, expected, capacity * width, 0.0f);

    float collidingPositions[] = {3.0f, 7.0f};
    pipes[0] = (NnByte *)collidingPositions;
    bool rejectedRuntimeDuplicate = false;
    try {
        shiftForward_F32_F32(1u, 0u, 2u, &context);
    } catch (const std::runtime_error &) {
        rejectedRuntimeDuplicate = true;
    }
    assert(rejectedRuntimeDuplicate);

    printPassed("shiftSlots");
}

void testTopk() {
    float x[] = {1.0f, 4.0f, 2.0f, 3.0f};
    std::vector<NnUint> topk(2);
    topk_F32(x, topk.data(), 4u, 2u);
    assert(topk[0] == 1u);
    assert(topk[1] == 3u);
    printPassed("testTopk");
}

static void referenceAttention(
    float *output,
    const float *query,
    const float *currentKey,
    const float *currentValue,
    const float *keyCache,
    const float *valueCache,
    const float *positions,
    NnUint prefixLength,
    NnUint queryLength,
    const NnDiffusionAttentionOpConfig *config)
{
    const NnUint qDim = config->nQueryHeads * config->headDim;
    const NnUint kvDim = config->nKvHeads * config->headDim;
    const NnUint group = config->globalNQueryHeads / config->globalNKvHeads;
    for (NnUint qi = 0u; qi < queryLength; qi++) {
        for (NnUint qh = 0u; qh < config->nQueryHeads; qh++) {
            const NnUint kvh = (config->queryHeadStart + qh) / group - config->kvHeadStart;
            std::vector<float> scores;
            std::vector<const float *> values;
            NnUint prefixStart = 0u;
            if (config->slidingWindow != 0u) {
                const NnUint end = config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL
                    ? (NnUint)positions[qi] + 1u
                    : prefixLength;
                prefixStart = end > config->slidingWindow ? end - config->slidingWindow : 0u;
            }
            NnUint prefixEnd = prefixLength;
            if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL)
                prefixEnd = std::min(prefixEnd, (NnUint)positions[qi] + 1u);
            const float *q = &query[qi * qDim + qh * config->headDim];
            for (NnUint p = prefixStart; p < prefixEnd; p++) {
                const NnUint slot = config->slidingWindow == 0u ? p : p % config->cacheCapacity;
                const float *k = &keyCache[slot * kvDim + kvh * config->headDim];
                scores.push_back(dotProduct_F32(q, k, config->headDim) * config->scoreScale);
                values.push_back(&valueCache[slot * kvDim + kvh * config->headDim]);
            }
            for (NnUint ki = 0u; ki < queryLength; ki++) {
                const NnUint kp = (NnUint)positions[ki];
                if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL && kp > (NnUint)positions[qi])
                    continue;
                if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL &&
                    config->slidingWindow != 0u && kp < prefixStart)
                    continue;
                const float *k = &currentKey[ki * kvDim + kvh * config->headDim];
                scores.push_back(dotProduct_F32(q, k, config->headDim) * config->scoreScale);
                values.push_back(&currentValue[ki * kvDim + kvh * config->headDim]);
            }
            softmax_F32(scores.data(), scores.size());
            float *out = &output[qi * qDim + qh * config->headDim];
            std::memset(out, 0, config->headDim * sizeof(float));
            for (NnUint i = 0u; i < scores.size(); i++)
                for (NnUint d = 0u; d < config->headDim; d++)
                    out[d] += scores[i] * values[i][d];
        }
    }
}

void testDiffusionAttention() {
    const NnUint queryLength = 2u;
    const NnUint headDim = 4u;
    const NnUint qHeads = 2u;
    const NnUint kvHeads = 1u;
    float query[queryLength * qHeads * headDim] = {
        1, 0, 0, 0,  0, 1, 0, 0,
        0, 0, 1, 0,  0, 0, 0, 1,
    };
    float currentKey[queryLength * kvHeads * headDim] = {
        0.3f, 0.1f, 0.2f, 0.0f,
        0.0f, 0.4f, 0.1f, 0.2f,
    };
    float currentValue[queryLength * kvHeads * headDim] = {
        3, 4, 5, 6,
        7, 8, 9, 10,
    };
    float keyCache[4u * kvHeads * headDim] = {
        0.1f, 0.2f, 0.0f, 0.3f,
        0.2f, 0.0f, 0.4f, 0.1f,
        0.4f, 0.1f, 0.3f, 0.2f,
        0.0f, 0.3f, 0.2f, 0.4f,
    };
    float valueCache[4u * kvHeads * headDim] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16,
    };
    float positions[] = {2.0f, 3.0f};
    float actual[queryLength * qHeads * headDim];
    float expected[queryLength * qHeads * headDim];

    NnDiffusionAttentionOpConfig config = {
        DIFFUSION_ATTENTION_ENCODER_CAUSAL,
        qHeads, kvHeads, qHeads, kvHeads, 0u, 0u, headDim,
        0u, 4u, 1.0f,
        0u, 0u, 0u, 0u, 0u, 0u
    };
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 2u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 2u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionCausal", actual, expected, queryLength * qHeads * headDim, 0.00001f);

    config.mode = DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 2u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 2u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionBidir", actual, expected, queryLength * qHeads * headDim, 0.00001f);

    config.mode = DIFFUSION_ATTENTION_ENCODER_CAUSAL;
    config.slidingWindow = 2u;
    config.cacheCapacity = 4u;
    positions[0] = 4.0f;
    positions[1] = 5.0f;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 4u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 4u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionCausalSliding", actual, expected, queryLength * qHeads * headDim, 0.00001f);

    config.mode = DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 4u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 4u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionBidirSliding", actual, expected, queryLength * qHeads * headDim, 0.00001f);

    config.slidingWindow = 4u;
    positions[0] = 10.0f;
    positions[1] = 11.0f;
    config.mode = DIFFUSION_ATTENTION_ENCODER_CAUSAL;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 10u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 10u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionCausalWrappedPrefix", actual, expected,
        queryLength * qHeads * headDim, 0.00001f);

    config.mode = DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 10u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 10u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionBidirWrappedPrefix", actual, expected,
        queryLength * qHeads * headDim, 0.00001f);

    config.slidingWindow = 0u;
    positions[0] = 0.0f;
    positions[1] = 1.0f;
    referenceAttention(expected, query, currentKey, currentValue, keyCache, valueCache,
        positions, 0u, queryLength, &config);
    diffusionAttention_F32(actual, query, currentKey, currentValue, keyCache, valueCache,
        positions, 0u, queryLength, &config, 1u, 0u);
    compare_F32("diffusionAttentionNoPrefix", actual, expected, queryLength * qHeads * headDim, 0.00001f);
}

template <bool quantized, NnUint vocab = 64u, NnUint hiddenSize = 32u,
    NnUint batches = 3u, NnUint threads = 2u>
void testTiedEmbedding(float temperature = 0.7f) {
    const NnUint physicalSize = hiddenSize + 8u;
    const NnUint tileSize = 2u;
    const NnUint tileCount = (vocab + tileSize - 1u) / tileSize;
    const NnUint nodeIndex = 1u;
    const NnUint nNodes = 2u;
    const float softcap = 2.5f;

    std::vector<float> hidden(batches * hiddenSize);
    std::vector<float> embedding(vocab * hiddenSize);
    for (NnUint batch = 0u; batch < batches; ++batch)
        for (NnUint h = 0u; h < hiddenSize; ++h)
            hidden[batch * hiddenSize + h] =
                (float)((int)((batch + 2u) * (h + 3u) % 17u) - 8) * 0.025f;
    for (NnUint token = 0u; token < vocab; ++token)
        for (NnUint h = 0u; h < hiddenSize; ++h)
            embedding[token * hiddenSize + h] =
                (float)((int)((token + 1u) * (h + 5u) % 19u) - 9) * 0.02f;

    std::vector<float> effectiveHidden(hidden);
    std::vector<float> effectiveEmbedding(embedding);
    std::vector<float> effectiveMixEmbedding(embedding);
    std::vector<float> transposed(vocab * hiddenSize);
    for (NnUint token = 0u; token < vocab; ++token)
        for (NnUint h = 0u; h < hiddenSize; ++h)
            transposed[h * vocab + token] = embedding[token * hiddenSize + h];
    std::vector<NnBlockQ80> quantHidden(batches * hiddenSize / Q80_BLOCK_SIZE);
    std::vector<NnBlockQ40> quantEmbedding(vocab * hiddenSize / Q40_BLOCK_SIZE);
    std::vector<NnBlockQ40> quantMixEmbedding(vocab * hiddenSize / Q40_BLOCK_SIZE);
    if (quantized) {
        quantizeF32toQ80(hidden.data(), quantHidden.data(), hidden.size(), 1u, 0u);
        dequantizeQ80toF32(quantHidden.data(), effectiveHidden.data(), hidden.size(), 1u, 0u);
        quantizeF32toQ40(embedding.data(), quantEmbedding.data(), embedding.size(), 1u, 0u);
        dequantizeQ40toF32(quantEmbedding.data(), effectiveEmbedding.data(), embedding.size(), 1u, 0u);
        std::vector<float> effectiveTransposed(vocab * hiddenSize);
        quantizeF32toQ40(transposed.data(), quantMixEmbedding.data(), transposed.size(), 1u, 0u);
        dequantizeQ40toF32(quantMixEmbedding.data(), effectiveTransposed.data(),
            effectiveTransposed.size(), 1u, 0u);
        for (NnUint token = 0u; token < vocab; ++token)
            for (NnUint h = 0u; h < hiddenSize; ++h)
                effectiveMixEmbedding[token * hiddenSize + h] =
                    effectiveTransposed[h * vocab + token];
    }

    float rawLogits[batches][vocab];
    float logits[batches][vocab];
    float partials[batches][tileCount * physicalSize];
    float mix[batches][physicalSize];
    float probabilities[batches][vocab];
    float numerators[batches][hiddenSize];
    float maxima[batches * nNodes];
    std::fill(maxima, maxima + batches * nNodes, -123.0f);
    NnByte *logitRows[batches];
    NnByte *rawLogitRows[batches];
    NnByte *partialRows[batches];
    NnByte *maximumRows[batches];
    for (NnUint batch = 0u; batch < batches; ++batch) {
        for (NnUint token = 0u; token < vocab; ++token)
            rawLogits[batch][token] = dotProduct_F32(
                effectiveHidden.data() + batch * hiddenSize,
                effectiveEmbedding.data() + token * hiddenSize,
                hiddenSize);
        logitRows[batch] = reinterpret_cast<NnByte *>(logits[batch]);
        rawLogitRows[batch] = reinterpret_cast<NnByte *>(rawLogits[batch]);
        partialRows[batch] = reinterpret_cast<NnByte *>(partials[batch]);
        maximumRows[batch] = reinterpret_cast<NnByte *>(
            &maxima[batch * nNodes + nodeIndex]);
    }

    auto work = [&](NnUint thread) {
        SPLIT_THREADS(batchStart, batchEnd, batches, threads, thread);
        tiedEmbeddingPartial_F32(
            logitRows, rawLogitRows, &probabilities[0][0], &partials[0][0],
            vocab, hiddenSize, physicalSize, tileCount,
            1.0f / temperature, softcap, batchStart, batchEnd);
    };
    std::vector<std::thread> workers;
    for (NnUint thread = 1u; thread < threads; ++thread)
        workers.emplace_back(work, thread);
    work(0u);
    for (auto &worker : workers)
        worker.join();
    if (quantized) {
        NnBlockQ80 quantizedProbabilities[batches * vocab / Q80_BLOCK_SIZE];
        quantizeF32toQ80(&probabilities[0][0], quantizedProbabilities,
            batches * vocab, 1u, 0u);
        for (NnUint batch = 0u; batch < batches; ++batch)
            matmul_Q80_Q40_F32(numerators[batch],
                quantizedProbabilities + batch * vocab / Q80_BLOCK_SIZE,
                quantMixEmbedding.data(), vocab, hiddenSize, 1u, 0u);
    } else {
        for (NnUint batch = 0u; batch < batches; ++batch)
            matmul_F32_F32_F32(numerators[batch], probabilities[batch],
                transposed.data(), vocab, hiddenSize, 1u, 0u);
    }
    for (NnUint thread = 0u; thread < threads; ++thread) {
        SPLIT_THREADS(batchStart, batchEnd, batches, threads, thread);
        tiedEmbeddingReduce_F32(
            maximumRows, partialRows, &mix[0][0], &numerators[0][0],
            hiddenSize, physicalSize, tileCount,
            batchStart, batchEnd);
    }

    for (NnUint batch = 0u; batch < batches; ++batch) {
        float expectedLogits[vocab];
        float maximum = -INFINITY;
        for (NnUint token = 0u; token < vocab; ++token) {
            const float raw = dotProduct_F32(
                effectiveHidden.data() + batch * hiddenSize,
                effectiveEmbedding.data() + token * hiddenSize,
                hiddenSize);
            expectedLogits[token] = softcap * tanhf(raw / softcap);
            maximum = std::max(maximum, expectedLogits[token] / temperature);
        }
        compare_F32(quantized ? "tiedEmbeddingLogitsQ40" : "tiedEmbeddingLogitsF32",
            logits[batch], expectedLogits, vocab, 0.00001f);
        compare_F32(quantized ? "tiedEmbeddingMaxQ40" : "tiedEmbeddingMaxF32",
            &maxima[batch * nNodes + nodeIndex], &maximum, 1u,
            0.00001f + std::fabs(maximum) * 0.000001f);

        float probabilities[vocab];
        float effectiveProbabilities[vocab];
        float denominator = 0.0f;
        for (NnUint token = 0u; token < vocab; ++token) {
            probabilities[token] = expf(expectedLogits[token] / temperature - maximum);
            denominator += probabilities[token];
        }
        if (quantized) {
            NnBlockQ80 quantizedProbabilities[vocab / Q80_BLOCK_SIZE];
            quantizeF32toQ80(probabilities, quantizedProbabilities, vocab, 1u, 0u);
            dequantizeQ80toF32(quantizedProbabilities, effectiveProbabilities, vocab, 1u, 0u);
        } else {
            std::copy(probabilities, probabilities + vocab, effectiveProbabilities);
        }
        float actualMix[hiddenSize];
        float expectedMix[hiddenSize];
        for (NnUint h = 0u; h < hiddenSize; ++h) {
            float expected = 0.0f;
            for (NnUint token = 0u; token < vocab; ++token)
                expected += effectiveProbabilities[token] *
                    effectiveMixEmbedding[token * hiddenSize + h];
            expectedMix[h] = expected / denominator;
            actualMix[h] = mix[batch][h] / mix[batch][hiddenSize];
        }
        compare_F32(quantized ? "tiedEmbeddingMixQ40" : "tiedEmbeddingMixF32",
            actualMix, expectedMix, hiddenSize, 0.00001f);
        if (quantized) {
            for (NnUint h = 0u; h < hiddenSize; ++h) {
                float canonical = 0.0f;
                for (NnUint token = 0u; token < vocab; ++token)
                    canonical += probabilities[token] *
                        effectiveEmbedding[token * hiddenSize + h];
                assert(std::fabs(expectedMix[h] - canonical / denominator) < 0.025f);
            }
        }
        for (NnUint h = hiddenSize + 1u; h < physicalSize; ++h)
            assert(mix[batch][h] == 0.0f);
    }
}

void testTiedEmbeddingPartialTail() {
    const NnUint vocabSize = 67u;
    const NnUint hiddenSize = 3u;
    const NnUint physicalSize = 8u;
    const float inverseTemperature = 1.0f / 0.7f;
    const float softcap = 5.0f;
    float rawLogits[vocabSize];
    float logits[vocabSize];
    float probabilities[vocabSize];
    float partials[physicalSize];
    NnByte *logitRows[] = {reinterpret_cast<NnByte *>(logits)};
    NnByte *rawLogitRows[] = {reinterpret_cast<NnByte *>(rawLogits)};
    for (NnUint i = 0u; i < vocabSize; ++i)
        rawLogits[i] = (static_cast<int>(i) - 33) * 0.625f;

    tiedEmbeddingPartial_F32(logitRows, rawLogitRows, probabilities, partials,
        vocabSize, hiddenSize, physicalSize, 1u, inverseTemperature, softcap, 0u, 1u);

    float expectedLogits[vocabSize];
    float expectedProbabilities[vocabSize];
    float maximum = -INFINITY;
    for (NnUint i = 0u; i < vocabSize; ++i) {
        expectedLogits[i] = softcap * tanhf(rawLogits[i] / softcap);
        maximum = std::max(maximum, expectedLogits[i] * inverseTemperature);
    }
    float denominator = 0.0f;
    for (NnUint i = 0u; i < vocabSize; ++i) {
        expectedProbabilities[i] = expf(expectedLogits[i] * inverseTemperature - maximum);
        denominator += expectedProbabilities[i];
    }
    compare_F32("tiedEmbeddingPartialTailLogits", logits, expectedLogits, vocabSize, 0.00001f);
    compare_F32("tiedEmbeddingPartialTailProbabilities", probabilities,
        expectedProbabilities, vocabSize, 0.000002f);
    compare_F32("tiedEmbeddingPartialTailDenominator", partials + hiddenSize, &denominator,
        1u, 0.00002f);
    compare_F32("tiedEmbeddingPartialTailMaximum", partials + hiddenSize + 1u,
        &maximum, 1u, 0.00001f);
}

void testPaddedShards() {
    const NnUint nodes[] = {1u, 2u, 4u, 8u};
    const NnUint expectedMoe[] = {704u, 352u, 192u, 96u};
    const NnUint expectedDense[] = {2112u, 1056u, 544u, 288u};
    for (NnUint i = 0u; i < 4u; i++) {
        const NnPaddedShard moe = paddedShard(704u, nodes[i], nodes[i] - 1u, Q40_BLOCK_SIZE);
        const NnPaddedShard dense = paddedShard(2112u, nodes[i], nodes[i] - 1u, Q40_BLOCK_SIZE);
        assert(moe.physicalSize == expectedMoe[i]);
        assert(dense.physicalSize == expectedDense[i]);
        assert(moe.logicalEnd == 704u);
        assert(dense.logicalEnd == 2112u);
    }
    printPassed("paddedShards");
}

void testProportionalRope() {
    NnRopeSlice slice = {};
    slice.seqLen = 2u;
    slice.headDim = 4u;
    slice.qDim0 = 4u;
    slice.kvDim0 = 4u;
    slice.cacheSize = size2D(F_32, slice.seqLen, slice.headDim);
    NnRopeOpConfig config = {};
    config.type = ROPE_PROPORTIONAL;
    config.ropeScalingFactor = 1.0f;
    config.rotaryDim = 2u;
    config.slice = slice;
    config.slice.ropeTheta = 10000.0f;
    float cache[8];
    fullfillRopeCache(&config, cache);
    assert(std::fabs(cache[4] - std::cos(1.0f)) < 0.000001f);
    assert(std::fabs(cache[5] - 1.0f) < 0.000001f);
    assert(std::fabs(cache[6] - std::sin(1.0f)) < 0.000001f);
    assert(std::fabs(cache[7]) < 0.000001f);

    float value[] = {1.0f, 2.0f, 3.0f, 4.0f};
    ropeFalcon_F32(value, cache, true, 1u, &config.slice, 1u, 0u);
    float expected[] = {
        std::cos(1.0f) - 3.0f * std::sin(1.0f),
        2.0f,
        std::sin(1.0f) + 3.0f * std::cos(1.0f),
        4.0f,
    };
    compare_F32("proportionalPartialRope", value, expected, 4u, 0.000001f);
}

void testScaledShardedEmbedding() {
    float token = 11.0f;
    float weight[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float output[4];
    NnByte *inputs[] = {(NnByte *)&token};
    NnByte *outputs[] = {(NnByte *)output};
    NnEmbeddingOpConfig config = {10u, 2u, 2.0f};
    NnCpuOpContext context = {};
    context.opConfig = &config;
    context.input = inputs;
    context.output = outputs;
    context.weight = (NnByte *)weight;
    context.weightSize = size2D(F_32, 2u, 4u);
    context.outputSize = size2D(F_32, 1u, 4u);
    embeddingForward_F32_F32_F32(1u, 0u, 1u, &context);
    float expected[] = {10, 12, 14, 16};
    compare_F32("scaledShardedEmbedding", output, expected, 4u, 0.000001f);

    token = 3.0f;
    embeddingForward_F32_F32_F32(1u, 0u, 1u, &context);
    float zeros[] = {0, 0, 0, 0};
    compare_F32("shardedEmbeddingMiss", output, zeros, 4u, 0.000001f);

    const NnUint q40Hidden = 160u;
    std::vector<float> q40Weight(2u * q40Hidden);
    for (NnUint i = 0u; i < q40Weight.size(); ++i)
        q40Weight[i] = (float)((int)(i % 19u) - 9) * 0.125f;
    std::vector<NnBlockQ40> q40Blocks(q40Weight.size() / Q40_BLOCK_SIZE);
    quantizeF32toQ40(q40Weight.data(), q40Blocks.data(), q40Weight.size(), 1u, 0u);
    std::vector<float> q40Output(q40Hidden, -1234.0f);
    std::vector<float> q40Expected(q40Hidden);
    dequantizeQ40toF32(
        q40Blocks.data() + q40Hidden / Q40_BLOCK_SIZE,
        q40Expected.data(), q40Hidden, 1u, 0u);
    for (NnUint i = 0u; i < q40Hidden; ++i)
        q40Expected[i] *= 2.0f;
    NnByte *q40Outputs[] = {(NnByte *)q40Output.data()};
    context.output = q40Outputs;
    context.weight = (NnByte *)q40Blocks.data();
    context.weightSize = size2D(F_Q40, 2u, q40Hidden);
    context.outputSize = size2D(F_32, 1u, q40Hidden);
    token = 11.0f;
    // Five blocks split across four threads gives dequantization contiguous
    // ownership that differs from the old striped scaling pass. Running the
    // thread callbacks in reverse makes the former overwrite race
    // deterministic without relying on scheduler timing.
    for (NnUint thread = 4u; thread-- > 0u;)
        embeddingForward_F32_Q40_F32(4u, thread, 1u, &context);
    compare_F32("scaledShardedEmbeddingQ40", q40Output.data(),
        q40Expected.data(), q40Hidden, 0.000001f);
}

void testMultiPlaneCast() {
    float input[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f,
        10.0f, 11.0f, 12.0f,
    };
    float output[14];
    std::fill(output, output + 14, -1234.0f);
    NnByte *inputs[] = {
        (NnByte *)(input + 0),
        (NnByte *)(input + 3),
        (NnByte *)(input + 6),
        (NnByte *)(input + 9),
    };
    NnByte *outputs[] = {
        (NnByte *)(output + 0),
        (NnByte *)(output + 3),
        (NnByte *)(output + 6),
        (NnByte *)(output + 9),
    };
    NnCpuOpContext context = {};
    context.input = inputs;
    context.output = outputs;
    context.inputSize = size3D(F_32, 2u, 2u, 3u);
    context.outputSize = size3D(F_32, 2u, 2u, 3u);

    castForward_ANY(1u, 0u, 2u, &context);

    compare_F32("multiPlaneCast", input, output, 12u, 0.0f);
    assert(output[12] == -1234.0f);
    assert(output[13] == -1234.0f);
}

void testVocabOps() {
    const NnUint batchCount = 4u;
    const NnUint vocabSize = 8u;
    const NnUint microshardCount = 2u;
    const NnUint statsSize = microshardCount * NN_VOCAB_STATS_WIDTH;
    const NnUint vocabStart = 10u;
    float logits[batchCount * vocabSize];
    float globalMax[batchCount];
    float targetTokens[batchCount] = {10.0f, 13.0f, 16.0f, 17.0f};
    float samplingCoins[batchCount] = {0.0f, 0.25f, 0.75f, 0.999f};
    float stats[batchCount * statsSize];
    float samples[batchCount];
    const float temperature = 0.75f;
    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        for (NnUint token = 0u; token < vocabSize; ++token)
            logits[batch * vocabSize + token] = (float)token * 0.2f + (float)batch * 0.1f;
        globalMax[batch] = logits[batch * vocabSize + vocabSize - 1u] / temperature;
    }
    std::fill(stats, stats + batchCount * statsSize, -1234.0f);
    std::fill(samples, samples + batchCount, -1234.0f);

    NnByte *pipes[] = {
        (NnByte *)&temperature,
        (NnByte *)targetTokens,
        (NnByte *)samplingCoins,
        (NnByte *)logits,
    };
    NnPipeConfig pipeConfigs[] = {
        {NULL, size1D(F_32, 1u)},
        {NULL, size2D(F_32, batchCount, 1u)},
        {NULL, size2D(F_32, batchCount, 1u)},
        {NULL, size2D(F_32, batchCount, vocabSize)},
    };
    NnByte *buffers[] = {(NnByte *)globalMax};
    NnBufferConfig bufferConfigs[] = {{NULL, size2D(F_32, batchCount, 1u)}};
    NnByte *logitRows[batchCount];
    NnByte *statsRows[batchCount];
    NnByte *sampleRows[batchCount];
    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        logitRows[batch] = (NnByte *)(logits + batch * vocabSize);
        statsRows[batch] = (NnByte *)(stats + batch * statsSize);
        sampleRows[batch] = (NnByte *)(samples + batch);
    }

    NnVocabLocalStatsOpConfig localConfig = {0u, 0u, 1u, vocabStart, microshardCount, 1u};
    NnCpuOpContext localContext = {};
    localContext.nBatches = batchCount;
    localContext.buffers = buffers;
    localContext.bufferConfigs = bufferConfigs;
    localContext.pipes = pipes;
    localContext.pipeConfigs = pipeConfigs;
    localContext.opConfig = &localConfig;
    localContext.input = logitRows;
    localContext.inputSize = size2D(F_32, batchCount, vocabSize);
    localContext.output = statsRows;
    localContext.outputSize = size2D(F_32, batchCount, statsSize);
    localContext.weightSize = size0();
    initVocabLocalStatsForward(&localContext);
    for (NnUint thread = 3u; thread-- > 0u;)
        vocabLocalStatsForward(3u, thread, batchCount, &localContext);

    NnVocabSampleOpConfig sampleConfig = {0u, 0u, 2u, 3u, vocabStart, 0u, microshardCount};
    NnCpuOpContext sampleContext = localContext;
    sampleContext.opConfig = &sampleConfig;
    sampleContext.input = statsRows;
    sampleContext.inputSize = size2D(F_32, batchCount, statsSize);
    sampleContext.output = sampleRows;
    sampleContext.outputSize = size2D(F_32, batchCount, 1u);
    initVocabSampleForward(&sampleContext);
    for (NnUint thread = 3u; thread-- > 0u;)
        vocabSampleForward(3u, thread, batchCount, &sampleContext);

    float expectedSamples[batchCount];
    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        double total = 0.0;
        for (NnUint token = 0u; token < vocabSize; ++token)
            total += std::exp(shiftedVocabLogit(
                logits[batch * vocabSize + token], 1.0f / temperature, globalMax[batch]));
        const double threshold = samplingCoins[batch] * total;
        double cumulative = 0.0;
        NnUint sampled = vocabSize - 1u;
        for (NnUint token = 0u; token < vocabSize; ++token) {
            cumulative += std::exp(shiftedVocabLogit(
                logits[batch * vocabSize + token], 1.0f / temperature, globalMax[batch]));
            if (threshold < cumulative) {
                sampled = token;
                break;
            }
        }
        expectedSamples[batch] = (float)(vocabStart + sampled);
        for (NnUint shard = 0u; shard < microshardCount; ++shard) {
            const NnUint offset = batch * statsSize + shard * NN_VOCAB_STATS_WIDTH;
            const NnUint shardEnd = (shard + 1u) * (vocabSize / microshardCount);
            assert(stats[offset] > 0.0f && std::isfinite(stats[offset]));
            assert(stats[offset + 2u] == logits[batch * vocabSize + shardEnd - 1u]);
            assert(stats[offset + 3u] == (float)(vocabStart + shardEnd - 1u));
            const NnUint target = (NnUint)targetTokens[batch] - vocabStart;
            if (target >= shard * (vocabSize / microshardCount) && target < shardEnd)
                assert(stats[offset + 4u] == logits[batch * vocabSize + target]);
            else
                assert(std::isinf(stats[offset + 4u]) && stats[offset + 4u] < 0.0f);
        }
    }
    compare_F32("vocabOpsBatchParallel", samples, expectedSamples, batchCount, 0.0f);
}

void testVocabLocalStatsVectorTail() {
    const NnUint batchCount = 2u;
    const NnUint vocabSize = 34u;
    const NnUint microshardCount = 2u;
    const NnUint microshardSize = vocabSize / microshardCount;
    const NnUint statsSize = microshardCount * NN_VOCAB_STATS_WIDTH;
    const NnUint vocabStart = 100u;
    const float temperature = 0.7f;
    float logits[batchCount * vocabSize];
    float globalMax[batchCount];
    float targetTokens[batchCount] = {116.0f, 133.0f};
    float stats[batchCount * statsSize];
    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        for (NnUint token = 0u; token < vocabSize; ++token) {
            const int pattern = static_cast<int>((token * 19u + batch * 7u) % 31u) - 15;
            logits[batch * vocabSize + token] = static_cast<float>(pattern) * 0.37f;
        }
        // Exercise stable argmax tie handling in both a full vector and tail.
        logits[batch * vocabSize + 3u] = 9.0f + batch;
        logits[batch * vocabSize + 16u] = 9.0f + batch;
        logits[batch * vocabSize + 20u] = 10.0f + batch;
        logits[batch * vocabSize + 33u] = 10.0f + batch;
        globalMax[batch] = (10.0f + batch) / temperature;
    }

    NnByte *pipes[] = {(NnByte *)&temperature, (NnByte *)targetTokens};
    NnPipeConfig pipeConfigs[] = {
        {NULL, size1D(F_32, 1u)},
        {NULL, size2D(F_32, batchCount, 1u)},
    };
    NnByte *buffers[] = {(NnByte *)globalMax};
    NnBufferConfig bufferConfigs[] = {{NULL, size2D(F_32, batchCount, 1u)}};
    NnByte *logitRows[batchCount];
    NnByte *statsRows[batchCount];
    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        logitRows[batch] = (NnByte *)(logits + batch * vocabSize);
        statsRows[batch] = (NnByte *)(stats + batch * statsSize);
    }
    NnVocabLocalStatsOpConfig config = {0u, 0u, 1u, vocabStart, microshardCount, 1u};
    NnCpuOpContext context = {};
    context.nBatches = batchCount;
    context.buffers = buffers;
    context.bufferConfigs = bufferConfigs;
    context.pipes = pipes;
    context.pipeConfigs = pipeConfigs;
    context.opConfig = &config;
    context.input = logitRows;
    context.inputSize = size2D(F_32, batchCount, vocabSize);
    context.output = statsRows;
    context.outputSize = size2D(F_32, batchCount, statsSize);
    context.weightSize = size0();
    initVocabLocalStatsForward(&context);
    vocabLocalStatsForward(1u, 0u, batchCount, &context);

    for (NnUint batch = 0u; batch < batchCount; ++batch) {
        for (NnUint shard = 0u; shard < microshardCount; ++shard) {
            const NnUint start = shard * microshardSize;
            const NnUint end = start + microshardSize;
            double denominator = 0.0;
            double moment = 0.0;
            float maximum = -INFINITY;
            NnUint argmax = start;
            for (NnUint token = start; token < end; ++token) {
                const float logit = logits[batch * vocabSize + token];
                if (logit > maximum) {
                    maximum = logit;
                    argmax = token;
                }
                const double shifted = shiftedVocabLogit(
                    logit, 1.0f / temperature, globalMax[batch]);
                const double weight = std::exp(shifted);
                denominator += weight;
                moment += weight * shifted;
            }
            const float *record = stats + batch * statsSize + shard * NN_VOCAB_STATS_WIDTH;
            assert(std::fabs(record[0] - static_cast<float>(denominator)) < 0.00001f);
            assert(std::fabs(record[1] - static_cast<float>(moment)) < 0.00001f);
            assert(record[2] == maximum);
            assert(record[3] == static_cast<float>(vocabStart + argmax));
            const NnUint target = static_cast<NnUint>(targetTokens[batch]) - vocabStart;
            if (target >= start && target < end)
                assert(record[4] == logits[batch * vocabSize + target]);
            else
                assert(std::isinf(record[4]) && record[4] < 0.0f);
        }
    }
    printPassed("vocabLocalStatsVectorTail");
}

void testLearnedExpertScale() {
    float probabilities[] = {0.1f, 0.4f, 0.4f, 0.1f};
    float scales[] = {1.0f, 2.0f, 3.0f, 1.0f};
    NnUint indexes[2];
    float weights[2];
    routeExperts_F32(probabilities, 4u, 2u, true, scales, indexes, weights);
    assert(indexes[0] == 1u);
    assert(indexes[1] == 2u);
    assert(std::fabs(weights[0] - 1.0f) < 0.000001f);
    assert(std::fabs(weights[1] - 1.5f) < 0.000001f);
    printPassed("learnedExpertScale");
}

int main() {
    initQuants();

    printCpuInstructionSet();
    testSplitThreads();
    testConvertF32toF16();
    testQuantization(32);
    testQuantization(2);
    testQuantization(1);
    testInvRms();
    testRmsNorm(128);
    testScaleFreeRmsNorm();
    testMul(32);
    testMul(2);
    testMul(1);
    testAdd(32);
    testAdd(2);
    testAdd(1);
    testMergeSum();
    testSoftmax();
    testSilu();
    testMatmul_F32_Q40_F32(32);
    testMatmul_F32_Q40_F32(2);
    testMatmul_F32_Q40_F32(1);
    testLlamafileSgemm();
    testScale();
    testTanh();
    testShiftSlots();
    testTopk();
    testTiedEmbeddingPartialTail();
    testDiffusionAttention();
    testTiedEmbedding<false>();
    testTiedEmbedding<true>();
    // Multiple probability blocks, an odd hidden block count, and an uneven
    // thread split that exercises batched GEMM and its small-batch fallback.
    testTiedEmbedding<true, 256u, 96u, 13u, 3u>();
    testTiedEmbedding<true, 128u, 64u, 8u, 2u>();
    testTiedEmbedding<true, 256u, 96u, 13u, 3u>(0.001f);
    testTiedEmbedding<true, 32u, 32u, 3u, 8u>();
    testPaddedShards();
    testProportionalRope();
    testScaledShardedEmbedding();
    testMultiPlaneCast();
    testVocabOps();
    testVocabLocalStatsVectorTail();
    testLearnedExpertScale();
    return 0;
}
