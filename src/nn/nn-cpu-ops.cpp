#include <cmath>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <stdexcept>
#if defined(__ARM_NEON)
    #include <arm_neon.h>
#elif defined(__AVX2__) || defined(__AVX512F__)
    #include <immintrin.h>
#endif
#include "nn-cpu-ops.hpp"
#include "nn-quants.hpp"
#include "llamafile/sgemm.hpp"

#define DEBUG_OP_INPUT_OUTPUT false

#if DEBUG_OP_INPUT_OUTPUT
    #define DEBUG_VECTOR(context, suffix, v) \
        if (threadIndex == 0) { \
            printf("%20s.%6s: ", context->name, suffix); \
            for (int k = 0; k < 12; k++) printf("%f ", v[k]); \
            printf("\n"); \
        }

    #define DEBUG_SCALAR(context, suffix, scalar) \
        if (threadIndex == 0) \
            printf("%20s.%6s: %f\n", context->name, suffix, scalar);
#else
    #define DEBUG_VECTOR(context, suffix, vec)
    #define DEBUG_SCALAR(context, suffix, scalar)
#endif

#if defined(__ARM_NEON)
static inline float32x4_t expf_neon(float32x4_t x) {
    const float32x4_t ln2 = vdupq_n_f32(0.69314718056f);
    const float32x4_t inv_ln2 = vdupq_n_f32(1.44269504089f);
    const float32x4_t c1 = vdupq_n_f32(1.0f);
    const float32x4_t c2 = vdupq_n_f32(0.5f);
    const float32x4_t c3 = vdupq_n_f32(0.1666666667f);
    const float32x4_t c4 = vdupq_n_f32(0.04166666667f);
    const float32x4_t c5 = vdupq_n_f32(0.008333333333f);

    x = vminq_f32(x, vdupq_n_f32(88.0f));
    x = vmaxq_f32(x, vdupq_n_f32(-88.0f));

    float32x4_t kf = vaddq_f32(vmulq_f32(x, inv_ln2), vdupq_n_f32(0.5f));
    int32x4_t k = vcvtq_s32_f32(kf);
    kf = vcvtq_f32_s32(k);

    float32x4_t f = vmlsq_f32(x, kf, ln2);
    float32x4_t f2 = vmulq_f32(f, f);
    float32x4_t f3 = vmulq_f32(f2, f);
    float32x4_t f4 = vmulq_f32(f3, f);
    float32x4_t f5 = vmulq_f32(f4, f);
    float32x4_t p = c1;
    p = vaddq_f32(p, f);
    p = vaddq_f32(p, vmulq_f32(c2, f2));
    p = vaddq_f32(p, vmulq_f32(c3, f3));
    p = vaddq_f32(p, vmulq_f32(c4, f4));
    p = vaddq_f32(p, vmulq_f32(c5, f5));

    int32x4_t pow2k = vshlq_n_s32(vaddq_s32(k, vdupq_n_s32(127)), 23);
    float32x4_t two_k = vreinterpretq_f32_s32(pow2k);
    return vmulq_f32(p, two_k);
}
#endif

#if defined(__AVX2__)
static inline float horizontalSum_avx2(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

static inline float horizontalMax_avx2(__m256 v) {
    __m128 v_low = _mm256_castps256_ps128(v);
    __m128 v_high = _mm256_extractf128_ps(v, 1);
    __m128 max128 = _mm_max_ps(v_low, v_high);
    __m128 max64 = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    __m128 max32 = _mm_max_ss(max64, _mm_shuffle_ps(max64, max64, _MM_SHUFFLE(1, 1, 1, 1)));
    return _mm_cvtss_f32(max32);
}

static inline __m256 expf_avx2(__m256 x) {
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.0f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.0f));

    const __m256 log2e = _mm256_set1_ps(1.4426950408889634f);
    const __m256 c0 = _mm256_set1_ps(1.0f);
    const __m256 c1 = _mm256_set1_ps(0.6931471805599453f);
    const __m256 c2 = _mm256_set1_ps(0.2402265069591007f);
    const __m256 c3 = _mm256_set1_ps(0.05550410866482158f);
    const __m256 c4 = _mm256_set1_ps(0.009618129107628477f);
    __m256 y = _mm256_mul_ps(x, log2e);
    __m256i n = _mm256_cvtps_epi32(y);
    __m256 n_float = _mm256_cvtepi32_ps(n);
    __m256 f = _mm256_sub_ps(y, n_float);
    __m256 p = c4;
    p = _mm256_fmadd_ps(p, f, c3);
    p = _mm256_fmadd_ps(p, f, c2);
    p = _mm256_fmadd_ps(p, f, c1);
    p = _mm256_fmadd_ps(p, f, c0);
    __m256i exponent = _mm256_add_epi32(n, _mm256_set1_epi32(127));
    exponent = _mm256_slli_epi32(exponent, 23);
    __m256 two_n = _mm256_castsi256_ps(exponent);
    return _mm256_mul_ps(p, two_n);
}
#endif

static float invRms_F32(const float *x, const unsigned int size, const float epsilon) {
    float sum;
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fsq;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int j = 0; j < size; j += 4) {
        fsq = vld1q_f32(&x[j]);
        fs = vmlaq_f32(fs, fsq, fsq);
    }
    sum = vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int j = 0; j < size; j += 8) {
        a = _mm256_loadu_ps(&x[j]);
        u = _mm256_fmadd_ps(a, a, u);
    }
    sum = horizontalSum_avx2(u);
#else
    sum = 0;
    for (unsigned int j = 0; j < size; j++) {
        sum += x[j] * x[j];
    }
#endif
    sum /= size;
    sum += epsilon;
    return 1.0f / sqrtf(sum);
}

static void rmsNorm_F32(float *output, const float *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);
    float32x4_t fw;
    float32x4_t fx;
    float32x4_t fss = vmovq_n_f32(invRms);
    for (; i < neonEnd; i += 4) {
        fx = vld1q_f32(&x[i]);
        if (w != nullptr) {
            fw = vld1q_f32(&w[i]);
            fx = vmulq_f32(fx, fw);
        }
        fx = vmulq_f32(fx, fss);
        vst1q_f32(&output[i], fx);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    const __m256 invRmsVec = _mm256_set1_ps(invRms);
    for (; i < avxEnd; i += 8) {
        __m256 xVec = _mm256_loadu_ps(&x[i]);
        __m256 scaledX = _mm256_mul_ps(xVec, invRmsVec);
        __m256 result = scaledX;
        if (w != nullptr) {
            __m256 wVec = _mm256_loadu_ps(&w[i]);
            result = _mm256_mul_ps(scaledX, wVec);
        }
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++)
        output[i] = (w == nullptr ? 1.0f : w[i]) * (invRms * x[i]);
}

static void rmsNorm_Q80_F32_F32(float *output, const NnBlockQ80 *x, const float invRms, const float *w, const NnUint size, const NnUint nThreads, const NnUint threadIndex) {
    assert(size % Q80_BLOCK_SIZE == 0);
    const NnUint nBlocks = size / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

    for (NnUint i = start; i < end; i++) {
        float d = CONVERT_F16_TO_F32(x[i].d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            output[k] = (w == nullptr ? 1.0f : w[k]) * (invRms * d * x[i].qs[j]);
        }
    }
}

static void matmul_F32_F32_F32(float *output, const float *x, const float *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    unsigned int i, j;
#if defined(__ARM_NEON)
    assert(n % 4 == 0);
    float32x4_t q;
    float32x4_t p;
    float32x4_t z;
    for (i = start; i < end; i++) {
        z = vmovq_n_f32(0);
        for (j = 0; j < n; j += 4) {
            q = vld1q_f32(&x[j]);
            p = vld1q_f32(&w[i * n + j]);
            z = vfmaq_f32(z, q, p);
        }
        output[i] = vaddvq_f32(z);
    }
#elif defined(__AVX2__)
    assert(n % 8 == 0);
    __m256 a0, b0, u;
    for (i = start; i < end; i++) {
        u = _mm256_set1_ps(0.0f);
        for (j = 0; j < n; j += 8) {
            a0 = _mm256_loadu_ps(&x[j]);
            b0 = _mm256_loadu_ps(&w[i * n + j]);
            u = _mm256_fmadd_ps(a0, b0, u);
        }
        output[i] = horizontalSum_avx2(u);
    }
#else
    for (i = start; i < end; i++) {
        float val = 0.0f;
        for (j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        output[i] = val;
    }
#endif
}

static void matmul_Q80_Q40_F32(float *output, const NnBlockQ80 *x, const NnBlockQ40 *w, const NnUint n, const NnUint d, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, d, nThreads, threadIndex);
    assert(n % Q40_BLOCK_SIZE == 0);
    const unsigned int nBlocks = n / Q40_BLOCK_SIZE;

#if defined(__ARM_NEON)
    const uint8x16_t m4b = vdupq_n_u8(0x0F);
    const int8x16_t s8b = vdupq_n_s8(0x8);

    for (unsigned int di = start; di < end; di++) {
        float32x4_t sumv0 = vmovq_n_f32(0.0f);
        float32x4_t sumv1 = vmovq_n_f32(0.0f);
        float32x4_t sumv2 = vmovq_n_f32(0.0f);
        float32x4_t sumv3 = vmovq_n_f32(0.0f);

        unsigned int j = 0;
        
#if defined(__ARM_FEATURE_DOTPROD)
        for (; j + 3 < nBlocks; j += 4) {
            __builtin_prefetch(&w[di * nBlocks + j + 4]);
            __builtin_prefetch(&x[j + 4]);

            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ40 *w2 = &w[di * nBlocks + j + 2];
            const NnBlockQ40 *w3 = &w[di * nBlocks + j + 3];

            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];
            const NnBlockQ80 *x2 = &x[j + 2];
            const NnBlockQ80 *x3 = &x[j + 3];

            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w0->qs), m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w0->qs), 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w1->qs), m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w1->qs), 4)), s8b);
            int8x16_t w2l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w2->qs), m4b)), s8b);
            int8x16_t w2h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w2->qs), 4)), s8b);
            int8x16_t w3l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(vld1q_u8(w3->qs), m4b)), s8b);
            int8x16_t w3h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(vld1q_u8(w3->qs), 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);
            const int8x16_t x2l = vld1q_s8(x2->qs);
            const int8x16_t x2h = vld1q_s8(x2->qs + 16);
            const int8x16_t x3l = vld1q_s8(x3->qs);
            const int8x16_t x3h = vld1q_s8(x3->qs + 16);

            const int32x4_t p0 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w0l, x0l), w0h, x0h);
            const int32x4_t p1 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w1l, x1l), w1h, x1h);
            const int32x4_t p2 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w2l, x2l), w2h, x2h);
            const int32x4_t p3 = vdotq_s32(vdotq_s32(vdupq_n_s32(0), w3l, x3l), w3h, x3h);

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p0), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(p1), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
            sumv2 = vmlaq_n_f32(sumv2, vcvtq_f32_s32(p2), CONVERT_F16_TO_F32(w2->d) * CONVERT_F16_TO_F32(x2->d));
            sumv3 = vmlaq_n_f32(sumv3, vcvtq_f32_s32(p3), CONVERT_F16_TO_F32(w3->d) * CONVERT_F16_TO_F32(x3->d));
        }
#else
        for (; j + 1 < nBlocks; j += 2) {
            const NnBlockQ40 *w0 = &w[di * nBlocks + j];
            const NnBlockQ40 *w1 = &w[di * nBlocks + j + 1];
            const NnBlockQ80 *x0 = &x[j];
            const NnBlockQ80 *x1 = &x[j + 1];

            const uint8x16_t w0qs = vld1q_u8(w0->qs);
            const uint8x16_t w1qs = vld1q_u8(w1->qs);
            
            int8x16_t w0l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w0qs, m4b)), s8b);
            int8x16_t w0h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w0qs, 4)), s8b);
            int8x16_t w1l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(w1qs, m4b)), s8b);
            int8x16_t w1h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(w1qs, 4)), s8b);

            const int8x16_t x0l = vld1q_s8(x0->qs);
            const int8x16_t x0h = vld1q_s8(x0->qs + 16);
            const int8x16_t x1l = vld1q_s8(x1->qs);
            const int8x16_t x1h = vld1q_s8(x1->qs + 16);

            const int16x8_t pl0l = vmull_s8(vget_low_s8(w0l), vget_low_s8(x0l));
            const int16x8_t pl0h = vmull_s8(vget_high_s8(w0l), vget_high_s8(x0l));
            const int16x8_t ph0l = vmull_s8(vget_low_s8(w0h), vget_low_s8(x0h));
            const int16x8_t ph0h = vmull_s8(vget_high_s8(w0h), vget_high_s8(x0h));
            
            const int16x8_t pl1l = vmull_s8(vget_low_s8(w1l), vget_low_s8(x1l));
            const int16x8_t pl1h = vmull_s8(vget_high_s8(w1l), vget_high_s8(x1l));
            const int16x8_t ph1l = vmull_s8(vget_low_s8(w1h), vget_low_s8(x1h));
            const int16x8_t ph1h = vmull_s8(vget_high_s8(w1h), vget_high_s8(x1h));

            const int32x4_t pl0 = vaddq_s32(vpaddlq_s16(pl0l), vpaddlq_s16(pl0h));
            const int32x4_t ph0 = vaddq_s32(vpaddlq_s16(ph0l), vpaddlq_s16(ph0h));
            const int32x4_t pl1 = vaddq_s32(vpaddlq_s16(pl1l), vpaddlq_s16(pl1h));
            const int32x4_t ph1 = vaddq_s32(vpaddlq_s16(ph1l), vpaddlq_s16(ph1h));

            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(vaddq_s32(pl0, ph0)), CONVERT_F16_TO_F32(w0->d) * CONVERT_F16_TO_F32(x0->d));
            sumv1 = vmlaq_n_f32(sumv1, vcvtq_f32_s32(vaddq_s32(pl1, ph1)), CONVERT_F16_TO_F32(w1->d) * CONVERT_F16_TO_F32(x1->d));
        }
#endif

        for (; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[di * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];

            const uint8x16_t wqs = vld1q_u8(wb->qs);
            const int8x16_t wl = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(wqs, m4b)), s8b);
            const int8x16_t wh = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(wqs, 4)), s8b);

            const int8x16_t xl = vld1q_s8(xb->qs);
            const int8x16_t xh = vld1q_s8(xb->qs + 16);

#if defined(__ARM_FEATURE_DOTPROD)
            const int32x4_t p = vdotq_s32(vdotq_s32(vdupq_n_s32(0), wl, xl), wh, xh);
#else
            const int16x8_t pll = vmull_s8(vget_low_s8(wl), vget_low_s8(xl));
            const int16x8_t plh = vmull_s8(vget_high_s8(wl), vget_high_s8(xl));
            const int16x8_t phl = vmull_s8(vget_low_s8(wh), vget_low_s8(xh));
            const int16x8_t phh = vmull_s8(vget_high_s8(wh), vget_high_s8(xh));
            
            const int32x4_t pl = vaddq_s32(vpaddlq_s16(pll), vpaddlq_s16(plh));
            const int32x4_t ph = vaddq_s32(vpaddlq_s16(phl), vpaddlq_s16(phh));
            const int32x4_t p = vaddq_s32(pl, ph);
#endif
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            sumv0 = vmlaq_n_f32(sumv0, vcvtq_f32_s32(p), s);
        }

        output[di] = vaddvq_f32(sumv0) + vaddvq_f32(sumv1) + vaddvq_f32(sumv2) + vaddvq_f32(sumv3);
    }
#elif defined(__AVX512F__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w8 = _mm_loadu_si128((const __m128i*)wb->qs);
            __m128i v_w0 = _mm_and_si128(w8, _mm_set1_epi8(0x0F));
            __m128i v_w1 = _mm_srli_epi16(w8, 4);
            v_w1 = _mm_and_si128(v_w1, _mm_set1_epi8(0x0F));
            
            v_w0 = _mm_sub_epi8(v_w0, _mm_set1_epi8(8));
            v_w1 = _mm_sub_epi8(v_w1, _mm_set1_epi8(8));

            __m256i w8_combined = _mm256_set_m128i(v_w1, v_w0);
            __m512i w16 = _mm512_cvtepi8_epi16(w8_combined);

            __m256i x8 = _mm256_loadu_si256((const __m256i*)xb->qs);
            __m512i x16 = _mm512_cvtepi8_epi16(x8);

            __m512i products = _mm512_madd_epi16(w16, x16);
            sum += _mm512_reduce_add_epi32(products) * s;
        }
        output[i] = sum;
    }
#elif defined(__AVX2__)
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0f;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);

            __m128i w_packed = _mm_loadu_si128((const __m128i*)wb->qs);

            __m128i w0_low = _mm_and_si128(w_packed, _mm_set1_epi8(0x0F));
            __m128i w0 = _mm_sub_epi8(w0_low, _mm_set1_epi8(8));

            __m128i w1_high = _mm_srli_epi16(w_packed, 4);
            w1_high = _mm_and_si128(w1_high, _mm_set1_epi8(0x0F));
            __m128i w1 = _mm_sub_epi8(w1_high, _mm_set1_epi8(8));

            __m256i w0_16 = _mm256_cvtepi8_epi16(w0);
            __m256i w1_16 = _mm256_cvtepi8_epi16(w1);

            __m128i i1_8 = _mm_loadu_si128((const __m128i*)xb->qs);
            __m128i i2_8 = _mm_loadu_si128((const __m128i*)(xb->qs + Q40_BLOCK_SIZE / 2));

            __m256i i1_16 = _mm256_cvtepi8_epi16(i1_8);
            __m256i i2_16 = _mm256_cvtepi8_epi16(i2_8);

            __m256i prod0 = _mm256_mullo_epi16(w0_16, i1_16);
            __m256i prod1 = _mm256_mullo_epi16(w1_16, i2_16);
            __m256i sum_prod = _mm256_add_epi16(prod0, prod1);

            __m256i ones = _mm256_set1_epi16(1);
            __m256i sum32 = _mm256_madd_epi16(sum_prod, ones);

            __m128i sum_low = _mm256_castsi256_si128(sum32);
            __m128i sum_high = _mm256_extracti128_si256(sum32, 1);
            sum_low = _mm_add_epi32(sum_low, sum_high);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            sum_low = _mm_hadd_epi32(sum_low, sum_low);
            int32_t block_sum = _mm_extract_epi32(sum_low, 0);

            sum += block_sum * s;
        }
        output[i] = sum;
    }
#else
    for (NnUint i = start; i < end; i++) {
        float sum = 0.0;
        for (NnUint j = 0; j < nBlocks; j++) {
            const NnBlockQ40 *wb = &w[i * nBlocks + j];
            const NnBlockQ80 *xb = &x[j];
            const float s = CONVERT_F16_TO_F32(wb->d) * CONVERT_F16_TO_F32(xb->d);
            for (NnUint k = 0; k < Q40_BLOCK_SIZE / 2; k++) {
                const int w0 = (wb->qs[k] & 0x0F) - 8;
                const int w1 = (wb->qs[k] >> 4) - 8;
                const int i1 = xb->qs[k];
                const int i2 = xb->qs[k + Q80_BLOCK_SIZE / 2];
                sum += (w0 * i1 + w1 * i2) * s;
            }
        }
        output[i] = sum;
    }
#endif
}

#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f
#define GELU_COEF_A 0.044715f

static void gelu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        float x = output[i];
        output[i] = 0.5f * x * (1.0f + tanhf(SQRT_2_OVER_PI * x * (1.0f + GELU_COEF_A * x * x)));
    }
}

static void silu_F32(float *output, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;
#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 4);

    for (; i < neonEnd; i += 4) {
        float32x4_t x = vld1q_f32(&output[i]);
        float32x4_t neg_x = vnegq_f32(x);
        float32x4_t exp_negx = expf_neon(neg_x);
        float32x4_t denominator = vaddq_f32(exp_negx, vdupq_n_f32(1.0f));

        float32x4_t recip = vrecpeq_f32(denominator);
        recip = vmulq_f32(recip, vsubq_f32(vdupq_n_f32(2.0f), vmulq_f32(denominator, recip)));

        float32x4_t result = vmulq_f32(x, recip);
        vst1q_f32(output + i, result);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);

    const __m256 ones = _mm256_set1_ps(1.0f);
    const __m256 zero = _mm256_setzero_ps();
    for (; i < avxEnd; i += 8) {
        __m256 x_vec = _mm256_loadu_ps(output + i);
        __m256 neg_x = _mm256_sub_ps(zero, x_vec);
        __m256 exp_negx = expf_avx2(neg_x);
        __m256 denominator = _mm256_add_ps(ones, exp_negx);
        __m256 result = _mm256_div_ps(x_vec, denominator);
        _mm256_storeu_ps(output + i, result);
    }
#endif
    for (; i < end; i++) {
        float x = output[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

static void tanh_F32(float *output, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++)
        output[i] = tanhf(output[i]);
}

static void add_F32(float *output, const float *x, const unsigned int n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    for (unsigned int i = start; i < end; i++) {
        output[i] += x[i];
    }
}

static void add_Q80_F32(float *y, const NnBlockQ80 *x, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);

#if defined(__ARM_NEON)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        float *y_base = y + i * Q80_BLOCK_SIZE;
        const int8_t *qs = xi->qs;

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 16) {
            // Load 16x 8-bit quantized values
            const int8x16_t q8 = vld1q_s8(qs + j);
            
            // Split into 8-bit high/low components
            const int8x8_t q8_low = vget_low_s8(q8);
            const int8x8_t q8_high = vget_high_s8(q8);
            
            // Sign extend to 16-bit
            const int16x8_t q16_low = vmovl_s8(q8_low);
            const int16x8_t q16_high = vmovl_s8(q8_high);
            
            // Sign extend to 32-bit and convert to float
            const float32x4_t qf_ll = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_low)));
            const float32x4_t qf_lh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_low)));
            const float32x4_t qf_hl = vcvtq_f32_s32(vmovl_s16(vget_low_s16(q16_high)));
            const float32x4_t qf_hh = vcvtq_f32_s32(vmovl_s16(vget_high_s16(q16_high)));
            
            // Multiply by scale factor
            const float32x4_t sf_ll = vmulq_n_f32(qf_ll, xid);
            const float32x4_t sf_lh = vmulq_n_f32(qf_lh, xid);
            const float32x4_t sf_hl = vmulq_n_f32(qf_hl, xid);
            const float32x4_t sf_hh = vmulq_n_f32(qf_hh, xid);
            
            // Load existing y values
            float32x4_t y_ll = vld1q_f32(y_base + j);
            float32x4_t y_lh = vld1q_f32(y_base + j + 4);
            float32x4_t y_hl = vld1q_f32(y_base + j + 8);
            float32x4_t y_hh = vld1q_f32(y_base + j + 12);
            
            // Accumulate results
            y_ll = vaddq_f32(y_ll, sf_ll);
            y_lh = vaddq_f32(y_lh, sf_lh);
            y_hl = vaddq_f32(y_hl, sf_hl);
            y_hh = vaddq_f32(y_hh, sf_hh);
            
            // Store results back
            vst1q_f32(y_base + j, y_ll);
            vst1q_f32(y_base + j + 4, y_lh);
            vst1q_f32(y_base + j + 8, y_hl);
            vst1q_f32(y_base + j + 12, y_hh);
        }
    }
#elif defined(__AVX2__)
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);

        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j += 8) {
            __m128i i8_vec = _mm_loadl_epi64((const __m128i*)(xi->qs + j));

            __m256i i32_vec = _mm256_cvtepi8_epi32(i8_vec);

            __m256 f_vec = _mm256_cvtepi32_ps(i32_vec);

            __m256 scale = _mm256_set1_ps(xid);
            __m256 scaled = _mm256_mul_ps(f_vec, scale);

            float* y_ptr = y + i * Q80_BLOCK_SIZE + j;
            __m256 y_vec = _mm256_loadu_ps(y_ptr);
            y_vec = _mm256_add_ps(y_vec, scaled);
            _mm256_storeu_ps(y_ptr, y_vec);
        }
    }
#else
    for (unsigned int i = start; i < end; i++) {
        const NnBlockQ80 *xi = &x[i];
        const float xid = CONVERT_F16_TO_F32(xi->d);
        for (unsigned int j = 0; j < Q80_BLOCK_SIZE; j++) {
            y[i * Q80_BLOCK_SIZE + j] += xid * xi->qs[j];
        }
    }
#endif
}

void softmax_F32(float *x, const NnUint size) {
    if (size == 0)
        return;

#if defined(__ARM_NEON)
    NnUint j;
    float maxVal;
    if (size >= 4) {
        float32x4_t fs;
        float32x4_t fmaxv = vld1q_f32(&x[0]);
        j = size - (size % 4);
        for (NnUint i = 4; i < j; i += 4) {
            fs = vld1q_f32(&x[i]);
            fmaxv = vmaxq_f32(fmaxv, fs);
        }
        maxVal = vmaxvq_f32(fmaxv);
    } else {
        maxVal = x[0];
        j = 1;
    }
    for (; j < size; j++)
        maxVal = fmaxf(maxVal, x[j]);

    const float32x4_t maxVal_vec = vdupq_n_f32(maxVal);
    float32x4_t sumv = vdupq_n_f32(0.0f);
    NnUint i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vsubq_f32(val, maxVal_vec);
        val = expf_neon(val);
        vst1q_f32(x + i, val);
        sumv = vaddq_f32(sumv, val);
    }

    float32x2_t sum_lo = vadd_f32(vget_low_f32(sumv), vget_high_f32(sumv));
    float sum = vget_lane_f32(sum_lo, 0) + vget_lane_f32(sum_lo, 1);

    for (; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const float32x4_t inv_sum_vec = vdupq_n_f32(inv_sum);

    i = 0;
    for (; i + 4 <= size; i += 4) {
        float32x4_t val = vld1q_f32(x + i);
        val = vmulq_f32(val, inv_sum_vec);
        vst1q_f32(x + i, val);
    }
    for (; i < size; ++i)
        x[i] /= sum;
#elif defined(__AVX2__)
    float maxVal;
    const unsigned avxEnd = size - (size % 8);
    NnUint i = 0;

    if (avxEnd >= 8) {
        __m256 max_vec = _mm256_loadu_ps(x);
        i = 8;
        for (; i < avxEnd; i += 8) {
            __m256 vec = _mm256_loadu_ps(&x[i]);
            max_vec = _mm256_max_ps(max_vec, vec);
        }
        maxVal = horizontalMax_avx2(max_vec);
    } else {
        maxVal = x[0];
        i = 1;
    }
    for (; i < size; ++i) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }

    __m256 max_val_vec = _mm256_set1_ps(maxVal);
    __m256 sum_vec = _mm256_setzero_ps();
    float sum = 0.0f;
    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(&x[i]);
        vec = _mm256_sub_ps(vec, max_val_vec);
        vec = expf_avx2(vec);
        _mm256_storeu_ps(&x[i], vec);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    sum = horizontalSum_avx2(sum_vec);
    for (; i < size; ++i) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }

    if (sum == 0.0f)
        sum = 0.000001f;

    const float inv_sum = 1.0f / sum;
    const __m256 inv_sum_vec = _mm256_set1_ps(inv_sum);

    i = 0;
    for (; i < avxEnd; i += 8) {
        __m256 vec = _mm256_loadu_ps(x + i);
        vec = _mm256_mul_ps(vec, inv_sum_vec);
        _mm256_storeu_ps(x + i, vec);
    }
    for (; i < size; i++)
        x[i] *= inv_sum;
#else
    float maxVal = x[0];
    for (NnUint i = 1; i < size; i++) {
        if (x[i] > maxVal)
            maxVal = x[i];
    }
    float sum = 0.0f;
    for (NnUint i = 0; i < size; i++) {
        x[i] = expf(x[i] - maxVal);
        sum += x[i];
    }
    if (sum == 0.0)
        sum = 0.000001;
    for (NnUint i = 0; i < size; i++)
        x[i] /= sum;
#endif
}

static float dotProduct_F32(const float *a, const float *b, const unsigned int size) {
#if defined(__ARM_NEON)
    assert(size % 4 == 0);
    float32x4_t fa;
    float32x4_t fb;
    float32x4_t fs = vmovq_n_f32(0);
    for (unsigned int i = 0; i < size; i += 4) {
        fa = vld1q_f32(&a[i]);
        fb = vld1q_f32(&b[i]);
        fs = vmlaq_f32(fs, fa, fb);
    }
    return vaddvq_f32(fs);
#elif defined(__AVX2__)
    assert(size % 8 == 0);
    __m256 a0, b0;
    __m256 u = _mm256_set1_ps(0.0f);
    for (unsigned int i = 0; i < size; i += 8) {
        a0 = _mm256_loadu_ps(&a[i]);
        b0 = _mm256_loadu_ps(&b[i]);
        u = _mm256_fmadd_ps(a0, b0, u);
    }
    return horizontalSum_avx2(u);
#else
    float sum = 0.0f;
    for (unsigned int i = 0; i < size; i++) {
        sum += a[i] * b[i];
    }
    return sum;
#endif
}

static void multiheadAtt_F32(
    float *y, const float *q, float *att, float *keyCache, float *valueCache,
    const NnUint pos, const NnUint nHeads, const NnUint nHeads0, const NnUint nKvHeads, const NnUint kvDim0, const NnUint headDim, const NnUint seqLen,
    const NnUint nThreads, const NnUint threadIndex) 
{
    SPLIT_THREADS(h0Start, h0End, nHeads0, nThreads, threadIndex);
    const NnUint kvMul = nHeads / nKvHeads;
    const float headDimRoot = sqrtf(headDim);

    for (NnUint h0 = h0Start; h0 < h0End; h0++) {
        const float *hQ = &q[h0 * headDim];
        const NnUint headIndex = h0 / kvMul;
        const float *hKc = &keyCache[headIndex * headDim];
        const float *hVc = &valueCache[headIndex * headDim];
        float *hAtt = &att[h0 * seqLen];

        for (NnUint t = 0; t <= pos; t++) {
            const float *posK = &hKc[t * kvDim0];
            const float score = dotProduct_F32(hQ, posK, headDim) / headDimRoot;
            hAtt[t] = score;
        }

        softmax_F32(hAtt, pos + 1);

        float *hY = &y[h0 * headDim];
        std::memset(hY, 0, headDim * sizeof(float));

        for (NnUint t = 0; t <= pos; t++) {
            const float *posV = &hVc[t * kvDim0];
            const float posA = hAtt[t];
            for (int i = 0; i < headDim; i++) {
                hY[i] += posA * posV[i];
            }
        }
    }
}

static inline void onlineAttentionAccumulate(
    float *output,
    float *runningMax,
    float *runningSum,
    const float score,
    const float *value,
    const NnUint headDim)
{
    const float nextMax = std::max(*runningMax, score);
    const float oldScale = *runningSum == 0.0f ? 0.0f : expf(*runningMax - nextMax);
    const float newScale = expf(score - nextMax);
    for (NnUint d = 0u; d < headDim; d++)
        output[d] = output[d] * oldScale + value[d] * newScale;
    *runningSum = *runningSum * oldScale + newScale;
    *runningMax = nextMax;
}

static void diffusionAttention_F32(
    float *output,
    const float *query,
    const float *currentKey,
    const float *currentValue,
    const float *keyCache,
    const float *valueCache,
    const float *positions,
    const NnUint prefixLength,
    const NnUint queryLength,
    const NnDiffusionAttentionOpConfig *config,
    const NnUint nThreads,
    const NnUint threadIndex)
{
    const NnUint qDim = config->nQueryHeads * config->headDim;
    const NnUint kvDim = config->nKvHeads * config->headDim;
    const NnUint kvGroup = config->globalNQueryHeads / config->globalNKvHeads;
    const NnUint taskCount = queryLength * config->nQueryHeads;
    SPLIT_THREADS(taskStart, taskEnd, taskCount, nThreads, threadIndex);

    for (NnUint task = taskStart; task < taskEnd; task++) {
        const NnUint qIndex = task / config->nQueryHeads;
        const NnUint localQHead = task % config->nQueryHeads;
        const NnUint globalQHead = config->queryHeadStart + localQHead;
        const NnUint globalKvHead = globalQHead / kvGroup;
        assert(globalKvHead >= config->kvHeadStart);
        const NnUint localKvHead = globalKvHead - config->kvHeadStart;
        assert(localKvHead < config->nKvHeads);

        const float *q = &query[qIndex * qDim + localQHead * config->headDim];
        float *out = &output[qIndex * qDim + localQHead * config->headDim];
        std::memset(out, 0, config->headDim * sizeof(float));
        float runningMax = -INFINITY;
        float runningSum = 0.0f;
        const NnUint qPosition = (NnUint)positions[qIndex];

        NnUint prefixStart = 0u;
        if (config->slidingWindow != 0u) {
            const NnUint windowEnd = config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL
                ? qPosition + 1u
                : prefixLength;
            prefixStart = windowEnd > config->slidingWindow
                ? windowEnd - config->slidingWindow
                : 0u;
        }
        NnUint prefixEnd = prefixLength;
        if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL)
            prefixEnd = std::min(prefixEnd, qPosition + 1u);

        for (NnUint absolute = prefixStart; absolute < prefixEnd; absolute++) {
            const NnUint slot = config->slidingWindow == 0u
                ? absolute
                : absolute % config->cacheCapacity;
            const float *k = &keyCache[slot * kvDim + localKvHead * config->headDim];
            const float *v = &valueCache[slot * kvDim + localKvHead * config->headDim];
            const float score = dotProduct_F32(q, k, config->headDim) * config->scoreScale;
            onlineAttentionAccumulate(out, &runningMax, &runningSum, score, v, config->headDim);
        }

        for (NnUint keyIndex = 0u; keyIndex < queryLength; keyIndex++) {
            const NnUint keyPosition = (NnUint)positions[keyIndex];
            if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL && keyPosition > qPosition)
                continue;
            if (config->mode == DIFFUSION_ATTENTION_ENCODER_CAUSAL &&
                config->slidingWindow != 0u && keyPosition < prefixStart)
                continue;
            const float *k = &currentKey[keyIndex * kvDim + localKvHead * config->headDim];
            const float *v = &currentValue[keyIndex * kvDim + localKvHead * config->headDim];
            const float score = dotProduct_F32(q, k, config->headDim) * config->scoreScale;
            onlineAttentionAccumulate(out, &runningMax, &runningSum, score, v, config->headDim);
        }

        if (runningSum != 0.0f) {
            const float invSum = 1.0f / runningSum;
            for (NnUint d = 0u; d < config->headDim; d++)
                out[d] *= invSum;
        }
    }
}

static inline void dequantizeQ40Block_F32(
    const NnBlockQ40 *block,
    float *output,
    const float scale = 1.0f);

static void weightedAccumulate_F32(
    float *output,
    const float *value,
    const NnUint size,
    const float outputScale,
    const float valueScale)
{
    NnUint i = 0u;
#if defined(__ARM_NEON)
    const float32x4_t outputScaleVec = vdupq_n_f32(outputScale);
    const float32x4_t valueScaleVec = vdupq_n_f32(valueScale);
    for (; i + 4u <= size; i += 4u) {
        float32x4_t out = vld1q_f32(output + i);
        const float32x4_t row = vld1q_f32(value + i);
        if (outputScale != 1.0f)
            out = vmulq_f32(out, outputScaleVec);
        out = vfmaq_f32(out, row, valueScaleVec);
        vst1q_f32(output + i, out);
    }
#elif defined(__AVX2__)
    const __m256 outputScaleVec = _mm256_set1_ps(outputScale);
    const __m256 valueScaleVec = _mm256_set1_ps(valueScale);
    for (; i + 8u <= size; i += 8u) {
        __m256 out = _mm256_loadu_ps(output + i);
        const __m256 row = _mm256_loadu_ps(value + i);
        if (outputScale != 1.0f)
            out = _mm256_mul_ps(out, outputScaleVec);
        out = _mm256_fmadd_ps(row, valueScaleVec, out);
        _mm256_storeu_ps(output + i, out);
    }
#endif
    for (; i < size; ++i)
        output[i] = output[i] * outputScale + value[i] * valueScale;
}

static void tiedEmbeddingPartial_F32_F32_F32(
    NnByte **logitRows,
    NnByte **hiddenRows,
    const float *embedding,
    float *partials,
    const NnUint vocabSize,
    const NnUint hiddenSize,
    const NnUint physicalSize,
    const NnUint tileCount,
    const float inverseTemperature,
    const float logitSoftcap,
    const NnUint batchSize,
    const NnUint batchStart,
    const NnUint batchEnd)
{
    if (batchStart == batchEnd)
        return;
    std::vector<float> runningMax(batchSize);
    std::vector<float> denominator(batchSize);
    std::vector<float> outputScale(batchSize);
    std::vector<float> valueScale(batchSize);

    for (NnUint batch = batchStart; batch < batchEnd; ++batch) {
        float *partial = partials + batch * tileCount * physicalSize;
        std::fill(partial, partial + physicalSize, 0.0f);
        for (NnUint tile = 1u; tile < tileCount; ++tile) {
            partial[tile * physicalSize + hiddenSize] = 0.0f;
            partial[tile * physicalSize + hiddenSize + 1u] = -INFINITY;
        }
    }

    for (NnUint token = 0u; token < vocabSize; ++token) {
        const float *row = embedding + token * hiddenSize;
        for (NnUint batch = batchStart; batch < batchEnd; ++batch) {
            const float *hidden = reinterpret_cast<const float *>(hiddenRows[batch]);
            const float rawLogit = dotProduct_F32(hidden, row, hiddenSize);
            const float logit = logitSoftcap * tanhf(rawLogit / logitSoftcap);
            reinterpret_cast<float *>(logitRows[batch])[token] = logit;
            const float score = logit * inverseTemperature;
            if (token == 0u) {
                runningMax[batch] = score;
                denominator[batch] = 1.0f;
                outputScale[batch] = 0.0f;
                valueScale[batch] = 1.0f;
            } else if (score > runningMax[batch]) {
                const float scale = expf(runningMax[batch] - score);
                runningMax[batch] = score;
                denominator[batch] = denominator[batch] * scale + 1.0f;
                outputScale[batch] = scale;
                valueScale[batch] = 1.0f;
            } else {
                const float scale = expf(score - runningMax[batch]);
                denominator[batch] += scale;
                outputScale[batch] = 1.0f;
                valueScale[batch] = scale;
            }
        }
        for (NnUint batch = batchStart; batch < batchEnd; ++batch) {
            weightedAccumulate_F32(
                partials + batch * tileCount * physicalSize,
                row,
                hiddenSize,
                outputScale[batch],
                valueScale[batch]);
        }
    }

    for (NnUint batch = batchStart; batch < batchEnd; ++batch) {
        float *partial = partials + batch * tileCount * physicalSize;
        partial[hiddenSize] = denominator[batch];
        partial[hiddenSize + 1u] = runningMax[batch];
    }
}

static void packTiedEmbeddingPanel_Q40(
    const NnBlockQ40 *weights, float *transposed,
    NnUint rowBlocks, NnUint length, NnUint panelSize)
{
    for (NnUint block = 0u; block < rowBlocks; ++block) {
        for (NnUint t = 0u; t < panelSize; t += 4u) {
            float rows[4][Q40_BLOCK_SIZE];
            for (NnUint r = 0u; r < 4u; ++r) {
                if (t + r < length)
                    dequantizeQ40Block_F32(&weights[(t + r) * rowBlocks + block], rows[r]);
                else
                    std::fill(rows[r], rows[r] + Q40_BLOCK_SIZE, 0.0f);
            }
            for (NnUint h = 0u; h < Q40_BLOCK_SIZE; h += 4u) {
                float *out = transposed + (block * Q40_BLOCK_SIZE + h) * panelSize + t;
#if defined(__ARM_NEON)
                const float32x4x2_t a = vtrnq_f32(vld1q_f32(rows[0] + h), vld1q_f32(rows[1] + h));
                const float32x4x2_t b = vtrnq_f32(vld1q_f32(rows[2] + h), vld1q_f32(rows[3] + h));
                vst1q_f32(out, vcombine_f32(vget_low_f32(a.val[0]), vget_low_f32(b.val[0])));
                vst1q_f32(out + panelSize, vcombine_f32(vget_low_f32(a.val[1]), vget_low_f32(b.val[1])));
                vst1q_f32(out + panelSize * 2u, vcombine_f32(vget_high_f32(a.val[0]), vget_high_f32(b.val[0])));
                vst1q_f32(out + panelSize * 3u, vcombine_f32(vget_high_f32(a.val[1]), vget_high_f32(b.val[1])));
#elif defined(__AVX2__)
                __m128 a = _mm_loadu_ps(rows[0] + h), b = _mm_loadu_ps(rows[1] + h);
                __m128 c = _mm_loadu_ps(rows[2] + h), d = _mm_loadu_ps(rows[3] + h);
                _MM_TRANSPOSE4_PS(a, b, c, d);
                _mm_storeu_ps(out, a);
                _mm_storeu_ps(out + panelSize, b);
                _mm_storeu_ps(out + panelSize * 2u, c);
                _mm_storeu_ps(out + panelSize * 3u, d);
#else
                for (NnUint c = 0u; c < 4u; ++c)
                    for (NnUint r = 0u; r < 4u; ++r)
                        out[c * panelSize + r] = rows[r][h + c];
#endif
            }
        }
    }
}

// Keep the two matrix products in cache-sized vocabulary panels. The original
// online, one-token-at-a-time path cannot reuse the GEMM register tiles across
// canvas positions. Only this small decoded panel is transposed, never the
// shared embedding tensor. Thread ownership is still by batch, with no barriers.
static void tiedEmbeddingPartial_Q80_Q40_F32(
    NnByte **logitRows,
    NnByte **hiddenRows,
    const NnBlockQ40 *embedding,
    float *partials,
    const NnUint vocabSize,
    const NnUint hiddenSize,
    const NnUint physicalSize,
    const NnUint tileCount,
    const float inverseTemperature,
    const float logitSoftcap,
    const NnUint batchStart,
    const NnUint batchEnd)
{
    const NnUint count = batchEnd - batchStart;
    if (count == 0u)
        return;
    const NnUint panelSize = 128u;
    const NnUint rowBlocks = hiddenSize / Q40_BLOCK_SIZE;
    std::vector<NnBlockQ80> hidden(count * rowBlocks);
    std::vector<float> scores(count * panelSize);
    std::vector<float> transposed(hiddenSize * panelSize);
    std::vector<float> numerator(count * hiddenSize);
    std::vector<float> maxima(count, -INFINITY);
    std::vector<float> denominators(count, 0.0f);
    for (NnUint b = 0u; b < count; ++b) {
        std::memcpy(hidden.data() + b * rowBlocks, hiddenRows[batchStart + b],
            rowBlocks * sizeof(NnBlockQ80));
        float *partial = partials + (batchStart + b) * tileCount * physicalSize;
        std::fill(partial, partial + physicalSize, 0.0f);
        for (NnUint tile = 1u; tile < tileCount; ++tile) {
            partial[tile * physicalSize + hiddenSize] = 0.0f;
            partial[tile * physicalSize + hiddenSize + 1u] = -INFINITY;
        }
    }
    for (NnUint start = 0u; start < vocabSize; start += panelSize) {
        const NnUint length = std::min(panelSize, vocabSize - start);
        const NnBlockQ40 *weights = embedding + start * rowBlocks;
        if (!llamafile_sgemm(length, count, rowBlocks, weights, rowBlocks,
                hidden.data(), rowBlocks, scores.data(), panelSize,
                0, 1, 0, F_Q40, F_Q80, F_32)) {
            for (NnUint b = 0u; b < count; ++b)
                matmul_Q80_Q40_F32(scores.data() + b * panelSize,
                    hidden.data() + b * rowBlocks, weights, hiddenSize, length, 1u, 0u);
        }
        for (NnUint b = 0u; b < count; ++b) {
            float *probabilities = scores.data() + b * panelSize;
            float *logits = reinterpret_cast<float *>(logitRows[batchStart + b]);
            float maximum = maxima[b];
            for (NnUint t = 0u; t < length; ++t) {
                const float logit = logitSoftcap * tanhf(probabilities[t] / logitSoftcap);
                logits[start + t] = logit;
                probabilities[t] = logit * inverseTemperature;
                maximum = std::max(maximum, probabilities[t]);
            }
            const float scale = expf(maxima[b] - maximum);
            float *partial = partials + (batchStart + b) * tileCount * physicalSize;
            for (NnUint h = 0u; h < hiddenSize; ++h)
                partial[h] *= scale;
            denominators[b] *= scale;
            maxima[b] = maximum;
            float denominator = 0.0f;
            for (NnUint t = 0u; t < length; ++t) {
                probabilities[t] = expf(probabilities[t] - maximum);
                denominator += probabilities[t];
            }
            denominators[b] += denominator;
            std::fill(probabilities + length, probabilities + panelSize, 0.0f);
        }
        packTiedEmbeddingPanel_Q40(weights, transposed.data(), rowBlocks, length, panelSize);
        if (!llamafile_sgemm(hiddenSize, count, panelSize,
                transposed.data(), panelSize, scores.data(), panelSize,
                numerator.data(), hiddenSize, 0, 1, 0, F_32, F_32, F_32)) {
            for (NnUint b = 0u; b < count; ++b)
                matmul_F32_F32_F32(numerator.data() + b * hiddenSize,
                    scores.data() + b * panelSize, transposed.data(),
                    panelSize, hiddenSize, 1u, 0u);
        }
        for (NnUint b = 0u; b < count; ++b)
            weightedAccumulate_F32(
                partials + (batchStart + b) * tileCount * physicalSize,
                numerator.data() + b * hiddenSize, hiddenSize, 1.0f, 1.0f);
    }
    for (NnUint b = 0u; b < count; ++b) {
        float *partial = partials + (batchStart + b) * tileCount * physicalSize;
        partial[hiddenSize] = denominators[b];
        partial[hiddenSize + 1u] = maxima[b];
    }
}

static void tiedEmbeddingReduce_F32(
    NnByte **maximumRows,
    NnByte **partialRows,
    float *embeddingMix,
    const NnUint hiddenSize,
    const NnUint physicalSize,
    const NnUint tileCount,
    const NnUint batchStart,
    const NnUint batchEnd)
{
    for (NnUint batch = batchStart; batch < batchEnd; ++batch) {
        const float *partials = reinterpret_cast<const float *>(partialRows[batch]);
        float *output = embeddingMix + batch * physicalSize;
        std::fill(output, output + physicalSize, 0.0f);
        float maximum = -INFINITY;
        for (NnUint tile = 0u; tile < tileCount; ++tile)
            maximum = std::max(maximum, partials[tile * physicalSize + hiddenSize + 1u]);
        float denominator = 0.0f;
        for (NnUint tile = 0u; tile < tileCount; ++tile) {
            const float *partial = partials + tile * physicalSize;
            if (partial[hiddenSize] == 0.0f)
                continue;
            const float scale = expf(partial[hiddenSize + 1u] - maximum);
            denominator += partial[hiddenSize] * scale;
            weightedAccumulate_F32(output, partial, hiddenSize, 1.0f, scale);
        }
        output[hiddenSize] = denominator;
        reinterpret_cast<float *>(maximumRows[batch])[0] = maximum;
    }
}

static void mul_F32(float *y, const float *x, const float *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, n, nThreads, threadIndex);
    unsigned int i = start;

#if defined(__ARM_NEON)
    const unsigned int count = end - start;
    const unsigned int neonEnd = end - (count % 8);
    for (; i < neonEnd; i += 4) {
        float32x4_t out_vec = vld1q_f32(&x[i]);
        float32x4_t x_vec = vld1q_f32(&m[i]);
        float32x4_t res_vec = vmulq_f32(out_vec, x_vec);
        vst1q_f32(&y[i], res_vec);
    }
#elif defined(__AVX2__)
    const unsigned int count = end - start;
    const unsigned int avxEnd = end - (count % 8);
    for (; i < avxEnd; i += 8) {
        __m256 out_vec = _mm256_loadu_ps(&x[i]);
        __m256 x_vec = _mm256_loadu_ps(&m[i]);
        __m256 res_vec = _mm256_mul_ps(out_vec, x_vec);
        _mm256_storeu_ps(&y[i], res_vec);
    }
#endif
    for (; i < end; i++)
        y[i] = x[i] * m[i];
}

static void scale_F32(const float *i, float *o, const float s, NnSize size, NnUint nThreads, NnUint threadIndex) {
    for (NnUint x = threadIndex; x < size; x += nThreads)
        o[x] = i[x] * s;
}

static void mul_Q80_F32(float *y, const float *x, const NnBlockQ80 *m, const NnUint n, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint nBlocks = n / Q80_BLOCK_SIZE;
    SPLIT_THREADS(start, end, nBlocks, nThreads, threadIndex);
    for (NnUint i = start; i < end; i++) {
        const NnBlockQ80 *b = &m[i];
        float d = CONVERT_F16_TO_F32(b->d);
        for (NnUint j = 0; j < Q80_BLOCK_SIZE; j++) {
            NnUint k = i * Q80_BLOCK_SIZE + j;
            y[k] = x[k] * d * b->qs[j];
        }
    }
}

static void copy_UNK(NnByte *output, const NnByte *x, NnSize size, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);
    NnUint s = end - start;
    if (s != 0)
        std::memcpy(&output[start], &x[start], s);
}


static void ropeLlama_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    const NnUint dim0Half = (isQ ? slice->qDim0 : slice->kvDim0) / 2;
    const NnUint shift = isQ ? slice->qShift : 0;
    SPLIT_THREADS(s, e, dim0Half, nThreads, threadIndex);
    const NnUint iStart = s * 2;
    const NnUint iEnd = e * 2;

    const float *posCache = &cache[pos * slice->sliceDim + shift];

    for (NnUint i = iStart; i < iEnd; i += 2) {
        const float fcr = posCache[i];
        const float fci = posCache[i + 1];
        const float v0 = x[i];
        const float v1 = x[i + 1];

        float x0 = v0 * fcr - v1 * fci;
        float x1 = v0 * fci + v1 * fcr;
        x[i] = x0;
        x[i + 1] = x1;
    }
}

static void ropeFalcon_F32(float* x, const float *cache, bool isQ, const NnUint pos, const NnRopeSlice *slice, const NnUint nThreads, const NnUint threadIndex) {
    unsigned int dim0 =  isQ ? slice->qDim0 : slice->kvDim0;
    assert(dim0 % slice->headDim == 0);
    unsigned int nHeads0 = dim0 / slice->headDim;
    SPLIT_THREADS(h0s, h0e, nHeads0, nThreads, threadIndex);

    const float *posCache = &cache[pos * slice->headDim];

    for (unsigned int h = h0s; h < h0e; h++) {
        const unsigned int o = h * slice->headDim;
        for (unsigned int j = 0; j < slice->headDim / 2; j++) {
            const float fcr0 = posCache[j];
            const float fci0 = posCache[j + slice->headDim / 2];

            float q0 = x[o + j];
            float q1 = x[o + j + slice->headDim / 2];
            x[o + j] = q0 * fcr0 - q1 * fci0;
            x[o + j + slice->headDim / 2] = q0 * fci0 + q1 * fcr0;
        }
    }
}

static void mergeSum_F32(float **output, float **input, const NnSize size, const NnSize batchSize, const NnSize nBatches, const NnSize nZ, const NnUint nThreads, const NnUint threadIndex) {
    SPLIT_THREADS(start, end, size, nThreads, threadIndex);

    for (NnUint y = 0u; y < batchSize; y++) {
        for (NnUint x = start; x < end; x++) {
            float s = 0.0f;
            for (NnUint z = 0u; z < nZ; z++)
                s += input[y + z * nBatches][x];
            output[y][x] = s;
        }
    }
}

static void topk_F32(const float *x, NnUint *y, NnSize size, NnUint k) {
    assert(k <= size);
    assert(k > 0u);

    std::vector<NnUint> items(size);
    for (NnSize i = 0u; i < size; i++)
        items[i] = i;

    std::sort(items.begin(), items.end(),
        [&x](int a, int b) {
            return x[a] == x[b] ? a < b : x[a] > x[b];
        }
    );

    for (NnUint i = 0u; i < k; i++)
        y[i] = items[i];
}

static void routeExperts_F32(
    const float *probabilities,
    NnUint nExperts,
    NnUint k,
    bool normalizeTopK,
    const float *expertScale,
    NnUint *indexes,
    float *weights)
{
    topk_F32(probabilities, indexes, nExperts, k);
    float sum = 1.0f;
    if (normalizeTopK) {
        sum = 0.0f;
        for (NnUint i = 0u; i < k; i++)
            sum += probabilities[indexes[i]];
        if (!(sum > 0.0f) || !std::isfinite(sum))
            throw std::runtime_error("MOE_GATE normalized top-k sum must be finite and positive");
    }
    for (NnUint i = 0u; i < k; i++) {
        const NnUint expert = indexes[i];
        weights[i] = probabilities[expert] / sum * (expertScale == nullptr ? 1.0f : expertScale[expert]);
    }
}

//

static void mergeAddForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    NnUint nSlices = context->inputSize.x / context->outputSize.x;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        float *input = (float *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            float *i = &input[sliceIndex * context->outputSize.x];
            DEBUG_VECTOR(context, "input", i);
            add_F32(
                output,
                i,
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeAddForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->inputSize.floatType == F_Q80);
    assert(context->outputSize.floatType == F_32);

    NnUint nSlices = context->inputSize.x / context->outputSize.x;
    NnUint xSize = context->outputSize.x / Q80_BLOCK_SIZE;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *output = (float *)context->output[batchIndex];
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        for (NnUint sliceIndex = 0; sliceIndex < nSlices; sliceIndex++) {
            add_Q80_F32(
                output,
                &input[sliceIndex * xSize],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void mergeSumForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    const NnMergeSumOpCodeConfig *config = (const NnMergeSumOpCodeConfig *)context->opConfig;
    if (config->nGroups != 0u) {
        ASSERT_EQ(context->inputSize.z, 1u);
        ASSERT_EQ(context->outputSize.z, config->nGroups);
        assert(context->inputSize.x % (context->outputSize.x * config->nGroups) == 0u);
        const NnUint nContributions = context->inputSize.x / (context->outputSize.x * config->nGroups);
        for (NnUint group = 0u; group < config->nGroups; ++group) {
            for (NnUint batch = 0u; batch < batchSize; ++batch) {
                float *output = (float *)context->output[group * context->outputSize.y + batch];
                SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
                for (NnUint x = start; x < end; ++x) {
                    float sum = 0.0f;
                    const float *input = (const float *)context->input[batch];
                    for (NnUint contribution = 0u; contribution < nContributions; ++contribution)
                        sum += input[(contribution * config->nGroups + group) * context->outputSize.x + x];
                    output[x] = sum;
                }
            }
        }
        return;
    }
    ASSERT_EQ(context->outputSize.z, 1u);
    assert(context->inputSize.z >= 1u);

    mergeSum_F32(
        (float **)context->output,
        (float **)context->input,
        context->outputSize.x,
        batchSize,
        context->nBatches,
        context->inputSize.z,
        nThreads,
        threadIndex);
}

static inline float q80Value(const NnBlockQ80 *input, const NnUint index) {
    const NnBlockQ80 *block = &input[index / Q80_BLOCK_SIZE];
    return CONVERT_F16_TO_F32(block->d) * block->qs[index % Q80_BLOCK_SIZE];
}

static inline void dequantizeQ40Block_F32(
    const NnBlockQ40 *block,
    float *output,
    const float scale)
{
    const float blockScale = CONVERT_F16_TO_F32(block->d) * scale;
#if defined(__ARM_NEON)
    const uint8x16_t packed = vld1q_u8(block->qs);
    const int8x16_t offset = vdupq_n_s8(8);
    const int8x16_t low = vsubq_s8(
        vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0fu))), offset);
    const int8x16_t high = vsubq_s8(
        vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), offset);
    const float32x4_t blockScaleVec = vdupq_n_f32(blockScale);
    const int16x8_t low0 = vmovl_s8(vget_low_s8(low));
    const int16x8_t low1 = vmovl_s8(vget_high_s8(low));
    const int16x8_t high0 = vmovl_s8(vget_low_s8(high));
    const int16x8_t high1 = vmovl_s8(vget_high_s8(high));
    vst1q_f32(output,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low0))), blockScaleVec));
    vst1q_f32(output + 4u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(low0))), blockScaleVec));
    vst1q_f32(output + 8u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(low1))), blockScaleVec));
    vst1q_f32(output + 12u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(low1))), blockScaleVec));
    vst1q_f32(output + 16u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high0))), blockScaleVec));
    vst1q_f32(output + 20u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(high0))), blockScaleVec));
    vst1q_f32(output + 24u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(high1))), blockScaleVec));
    vst1q_f32(output + 28u,
        vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(high1))), blockScaleVec));
#elif defined(__AVX2__)
    const __m128i packed = _mm_loadu_si128(reinterpret_cast<const __m128i *>(block->qs));
    const __m128i low = _mm_sub_epi8(
        _mm_and_si128(packed, _mm_set1_epi8(0x0f)), _mm_set1_epi8(8));
    const __m128i high = _mm_sub_epi8(
        _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f)),
        _mm_set1_epi8(8));
    const __m256 blockScaleVec = _mm256_set1_ps(blockScale);
    _mm256_storeu_ps(output,
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(low)), blockScaleVec));
    _mm256_storeu_ps(output + 8u,
        _mm256_mul_ps(_mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(low, 8))), blockScaleVec));
    _mm256_storeu_ps(output + 16u,
        _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(high)), blockScaleVec));
    _mm256_storeu_ps(output + 24u,
        _mm256_mul_ps(_mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_srli_si128(high, 8))), blockScaleVec));
#else
    for (NnUint packedIndex = 0u; packedIndex < Q40_BLOCK_SIZE / 2u; ++packedIndex) {
        const NnUint packed = block->qs[packedIndex];
        output[packedIndex] = blockScale * (float)((int)(packed & 0x0fu) - 8);
        output[packedIndex + Q40_BLOCK_SIZE / 2u] =
            blockScale * (float)((int)(packed >> 4u) - 8);
    }
#endif
}

static void mergeSumForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    const NnMergeSumOpCodeConfig *config = (const NnMergeSumOpCodeConfig *)context->opConfig;
    if (config->nGroups != 0u) {
        ASSERT_EQ(context->inputSize.z, 1u);
        ASSERT_EQ(context->outputSize.z, config->nGroups);
        assert(context->inputSize.x % (context->outputSize.x * config->nGroups) == 0u);
        const NnUint nContributions = context->inputSize.x / (context->outputSize.x * config->nGroups);
        for (NnUint group = 0u; group < config->nGroups; ++group) {
            for (NnUint batch = 0u; batch < batchSize; ++batch) {
                float *output = (float *)context->output[group * context->outputSize.y + batch];
                const NnBlockQ80 *input = (const NnBlockQ80 *)context->input[batch];
                SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
                for (NnUint x = start; x < end; ++x) {
                    float sum = 0.0f;
                    for (NnUint contribution = 0u; contribution < nContributions; ++contribution) {
                        const NnUint index = (contribution * config->nGroups + group) * context->outputSize.x + x;
                        sum += q80Value(input, index);
                    }
                    output[x] = sum;
                }
            }
        }
        return;
    }
    ASSERT_EQ(context->outputSize.z, 1u);
    assert(context->inputSize.z >= 1u);
    for (NnUint batch = 0u; batch < batchSize; ++batch) {
        float *output = (float *)context->output[batch];
        SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
        for (NnUint x = start; x < end; ++x) {
            float sum = 0.0f;
            for (NnUint z = 0u; z < context->inputSize.z; ++z)
                sum += q80Value((const NnBlockQ80 *)context->input[batch + z * context->nBatches], x);
            output[x] = sum;
        }
    }
}

static void initEmbeddingForward(NnCpuOpContext *context) {
    const NnEmbeddingOpConfig *config = (NnEmbeddingOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.x, 1);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->weightSize.x, context->outputSize.x);
    const NnUint vocabSize = config->vocabSize == 0u ? context->weightSize.y : config->vocabSize;
    ASSERT_EQ(context->weightSize.y, vocabSize);
}

static NnUint embeddingLocalToken(const float tokenValue, const NnEmbeddingOpConfig *config, const NnUint vocabSize) {
    if (!std::isfinite(tokenValue) || tokenValue < 0.0f || std::floor(tokenValue) != tokenValue)
        throw std::runtime_error("EMBEDDING token must be a non-negative integer");
    const NnUint token = (NnUint)tokenValue;
    if (token < config->vocabStart || token >= config->vocabStart + vocabSize)
        return UINT32_MAX;
    return token - config->vocabStart;
}

static float embeddingScale(const NnEmbeddingOpConfig *config) {
    return config->scale == 0.0f ? 1.0f : config->scale;
}

static void embeddingForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnEmbeddingOpConfig *config = (NnEmbeddingOpConfig *)context->opConfig;
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnUint token = embeddingLocalToken(
            *((float *)context->input[batchIndex]), config, context->weightSize.y);
        float *output = (float *)context->output[batchIndex];
        if (token == UINT32_MAX) {
            SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
            std::fill(output + start, output + end, 0.0f);
        } else {
            const float *weight = (const float *)&context->weight[token * dimSize];
            scale_F32(weight, output, embeddingScale(config), context->outputSize.x, nThreads, threadIndex);
        }
    }
}

static void embeddingForward_F32_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnEmbeddingOpConfig *config = (NnEmbeddingOpConfig *)context->opConfig;
    const NnSize rowBytes = getBytes(F_Q40, context->outputSize.x);
    for (NnUint batchIndex = 0u; batchIndex < batchSize; batchIndex++) {
        const NnUint token = embeddingLocalToken(
            *((float *)context->input[batchIndex]), config, context->weightSize.y);
        float *output = (float *)context->output[batchIndex];
        if (token == UINT32_MAX) {
            SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
            std::fill(output + start, output + end, 0.0f);
        } else {
            const float scale = embeddingScale(config);
            const NnBlockQ40 *row =
                (const NnBlockQ40 *)&context->weight[token * rowBytes];
            const NnUint nBlocks = context->outputSize.x / Q40_BLOCK_SIZE;
            SPLIT_THREADS(blockStart, blockEnd, nBlocks, nThreads, threadIndex);
            for (NnUint block = blockStart; block < blockEnd; ++block) {
                const NnUint start = block * Q40_BLOCK_SIZE;
                dequantizeQ40Block_F32(&row[block], output + start, scale);
            }
        }
    }
}

static void embeddingForward_F32_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnEmbeddingOpConfig *config = (NnEmbeddingOpConfig *)context->opConfig;
    NnSize dimSize = getBytes(F_32, context->outputSize.x);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnUint token = embeddingLocalToken(
            *((float *)context->input[batchIndex]), config, context->weightSize.y);
        NnBlockQ80 *output = (NnBlockQ80 *)context->output[batchIndex];
        if (token == UINT32_MAX) {
            SPLIT_THREADS(blockStart, blockEnd, context->outputSize.x / Q80_BLOCK_SIZE, nThreads, threadIndex);
            std::memset(output + blockStart, 0, (blockEnd - blockStart) * sizeof(NnBlockQ80));
        } else {
            quantizeF32toQ80(
                (float *)&context->weight[token * dimSize],
                output,
                context->outputSize.x,
                nThreads,
                threadIndex);
            const float scale = embeddingScale(config);
            if (scale != 1.0f) {
                SPLIT_THREADS(blockStart, blockEnd, context->outputSize.x / Q80_BLOCK_SIZE, nThreads, threadIndex);
                for (NnUint block = blockStart; block < blockEnd; block++)
                    output[block].d = CONVERT_F32_TO_F16(CONVERT_F16_TO_F32(output[block].d) * scale);
            }
        }
    }
}

static void initInvRmsForward(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    assert(context->outputSize.x >= config->nColumns);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
}

static void invRmsForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnInvRmsOpConfig *config = (NnInvRmsOpConfig *)context->opConfig;
    const NnUint colSize = context->inputSize.x / config->nColumns;

    for (NnUint batchIndex = threadIndex; batchIndex < batchSize; batchIndex += nThreads) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            float rms = invRms_F32(
                &input[colIndex * colSize],
                colSize,
                config->epsilon);
            output[colIndex] = rms;
            DEBUG_SCALAR(context, "output", rms);
        }
    }
}

static void initRmsNormForward_ANY_F32_F32(NnCpuOpContext *context) {
    NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    NnBufferConfig *rmsBufferConfig = &context->bufferConfigs[config->invRmsBufferIndex];
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.x % config->nColumns, 0);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    if (config->scaleFree == 0u) {
        ASSERT_EQ(context->weightSize.floatType, F_32);
        ASSERT_EQ(context->weightSize.y, 1);
        ASSERT_EQ(context->weightSize.x, context->inputSize.x / config->nColumns);
    } else {
        assert(context->weightSize.nBytes == 0u);
    }
    ASSERT_EQ(rmsBufferConfig->size.floatType, F_32);
    assert(rmsBufferConfig->size.x >= config->nColumns);
    ASSERT_EQ(rmsBufferConfig->size.y, context->nBatches);
}

static void rmsNormForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    const float *weight = config->scaleFree == 0u ? (float *)context->weight : nullptr;
    const NnUint invRmsBatchSize = context->bufferConfigs[config->invRmsBufferIndex].size.x;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    const NnUint colSize = context->inputSize.x / config->nColumns;
    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *input = (float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        DEBUG_VECTOR(context, "input", input);
        for (NnUint colIndex = 0; colIndex < config->nColumns; colIndex++) {
            rmsNorm_F32(
                &output[colIndex * colSize],
                &input[colIndex * colSize],
                invRms[batchIndex * invRmsBatchSize + colIndex],
                weight,
                colSize,
                nThreads,
                threadIndex);
        }
        DEBUG_VECTOR(context, "output", output);
    }
}

static void rmsNormForward_Q80_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);

    const NnRmsNormOpConfig *config = (NnRmsNormOpConfig *)context->opConfig;
    ASSERT_EQ(config->nColumns, 1); // TODO: add support multiple columns

    const float *weight = config->scaleFree == 0u ? (float *)context->weight : nullptr;
    const float *invRms = (float *)context->buffers[config->invRmsBufferIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        NnBlockQ80 *input = (NnBlockQ80 *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        rmsNorm_Q80_F32_F32(
            output,
            input,
            invRms[batchIndex],
            weight,
            context->inputSize.x,
            nThreads,
            threadIndex);
        DEBUG_VECTOR(context, "output", output);
    }
}

static void initMatmulForward(NnCpuOpContext *context) {
    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->inputSize.x, context->weightSize.y);
    ASSERT_EQ(context->inputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->outputSize.x, context->weightSize.x);
    ASSERT_EQ(context->outputSize.z, std::max(config->nActiveExperts, 1u));
    ASSERT_EQ(context->weightSize.z, std::max(config->nExperts, 1u));

    if (!context->hasInputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for input\n", context->name);
    if (!context->hasOutputContinuousMemory)
        printf("🚧 Op %s does not have contiguous memory for output\n", context->name);

}

static bool matmulForward_llamafile(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (batchSize == 1u || !context->hasInputContinuousMemory || !context->hasOutputContinuousMemory || context->inputSize.z != 1u)
        return false;

    const NnUint n = context->weightSize.y / getBlockSize(context->inputSize.floatType);
    const NnUint d = context->weightSize.x;
    return llamafile_sgemm(
        d, batchSize, n,
        context->weight, n,
        context->input[0], n,
        context->output[0], d,
        threadIndex, nThreads, 0,
        context->weightSize.floatType,
        context->inputSize.floatType,
        F_32
    );
}

static void matmulForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            float *output = (float *)context->output[e * context->outputSize.y + y];
            matmul_F32_F32_F32(
                output,
                (float *)context->input[e * context->inputSize.y + y],
                (float *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY],
                context->weightSize.y,
                context->weightSize.x,
                nThreads,
                threadIndex);
            DEBUG_VECTOR(context, "output", output);
        }
    }
}

static void matmulForward_Q80_Q40_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    if (matmulForward_llamafile(nThreads, threadIndex, batchSize, context))
        return;

    const NnMatmulOpConfig *config = (NnMatmulOpConfig *)context->opConfig;
    const NnUint nActiveExpertsOr1 = std::max(config->nActiveExperts, 1u);
    const float *activeExpertIndexes = (const float *)context->buffers[config->activeExpertIndexesBufferIndex];

    for (NnUint y = 0; y < batchSize; y++) {
        for (NnUint e = 0; e < nActiveExpertsOr1; e++) {
            const NnUint activeExpertIndex = config->nActiveExperts == 0u
                ? 0u
                : (NnUint)activeExpertIndexes[y * config->nActiveExperts + e];

            float *output = (float *)context->output[e * context->outputSize.y + y];
            matmul_Q80_Q40_F32(
                output,
                (NnBlockQ80 *)context->input[e * context->inputSize.y + y],
                (NnBlockQ40 *)&context->weight[activeExpertIndex * context->weightSize.nBytesXY],
                context->weightSize.y,
                context->weightSize.x,
                nThreads,
                threadIndex);
            DEBUG_VECTOR(context, "output", output);
        }
    }
}

static void siluForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            float *output = (float *)context->output[z * context->outputSize.y + y];
            silu_F32(output, context->outputSize.x, nThreads, threadIndex);
        }
    }
}

static void tanhForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
    assert(*context->input == *context->output);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            float *output = (float *)context->output[z * context->outputSize.y + y];
            tanh_F32(output, context->outputSize.x, nThreads, threadIndex);
        }
    }
}

static void geluForward_F32_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);

    for (NnUint z = 0u; z < context->outputSize.z; ++z) {
        for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
            float *output = (float *)context->output[z * context->outputSize.y + batchIndex];
            gelu_F32(output, context->outputSize.x, nThreads, threadIndex);
        }
    }
}

static void initRopeForward_F32(NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    if (context->bufferFlags[config->ropeCacheBufferIndex] == 1)
        return;
    context->bufferFlags[config->ropeCacheBufferIndex] = 1;

    float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    fullfillRopeCache(config, cache);
}

static void ropeForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnRopeOpConfig *config = (NnRopeOpConfig *)context->opConfig;
    const NnRopeSlice *slice = &config->slice;
    const float *positions = (float *)context->pipes[config->positionPipeIndex];
    const float *cache = (float *)context->buffers[config->ropeCacheBufferIndex];
    const bool isQ = config->isQ == 1;

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *x = (float *)context->input[batchIndex];
        const NnUint pos = (NnUint)positions[batchIndex];
        if (config->type == ROPE_LLAMA || config->type == ROPE_LLAMA3_1)
            ropeLlama_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
        else if (config->type == ROPE_FALCON || config->type == ROPE_PROPORTIONAL)
            ropeFalcon_F32(x, cache, isQ, pos, slice, nThreads, threadIndex);
        else
            throw std::runtime_error("Unsupported rope type");
    }
}

static void initMultiHeadAttForward(NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    assert(context->weightSize.nBytes == 0);
    ASSERT_EQ(context->outputSize.x, config->qSliceD0);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    NnSize3D *querySize = &context->bufferConfigs[config->queryBufferIndex].size;
    ASSERT_EQ(querySize->x, config->qSliceD0);
    NnSize3D *posSize = &context->pipeConfigs[config->positionPipeIndex].size;
    ASSERT_EQ(posSize->x, 1);
    ASSERT_EQ(posSize->y, context->nBatches);
}

static void multiHeadAttForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMultiHeadAttOpConfig *config = (NnMultiHeadAttOpConfig *)context->opConfig;

    float *query = (float *)context->buffers[config->queryBufferIndex];
    float *keyCache = (float *)context->buffers[config->keyCacheBufferIndex];
    float *valueCache = (float *)context->buffers[config->valueCacheBufferIndex];
    float *att = (float *)context->buffers[config->attBufferIndex];
    const float *positions = (float *)context->pipes[config->positionPipeIndex];

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        float *y = (float *)context->output[batchIndex];
        float *q = &query[batchIndex * config->qSliceD0];
        NnUint pos = (NnUint)positions[batchIndex];
        assert(pos < config->seqLen);

        DEBUG_VECTOR(context, "input", y);
        DEBUG_VECTOR(context, "q", q);

        multiheadAtt_F32(y, q, 
            &att[batchIndex * config->nHeads0 * config->seqLen],
            keyCache, valueCache, pos,
            config->nHeads, config->nHeads0,
            config->nKvHeads, config->kvDim0, config->headDim, config->seqLen, nThreads, threadIndex);

        DEBUG_VECTOR(context, "output", y);
    }
}

static void initDiffusionAttentionForward(NnCpuOpContext *context) {
    const NnDiffusionAttentionOpConfig *config = (NnDiffusionAttentionOpConfig *)context->opConfig;
    assert(context->weightSize.nBytes == 0u);
    if (config->mode != DIFFUSION_ATTENTION_ENCODER_CAUSAL &&
        config->mode != DIFFUSION_ATTENTION_DECODER_BIDIRECTIONAL)
        throw std::invalid_argument("DIFFUSION_ATTENTION has an invalid mode");
    if (config->nQueryHeads == 0u || config->nKvHeads == 0u || config->headDim == 0u)
        throw std::invalid_argument("DIFFUSION_ATTENTION dimensions must be nonzero");
    if (config->globalNQueryHeads % config->globalNKvHeads != 0u)
        throw std::invalid_argument("DIFFUSION_ATTENTION requires an integral GQA ratio");
    if (config->slidingWindow != 0u && config->cacheCapacity == 0u)
        throw std::invalid_argument("DIFFUSION_ATTENTION sliding cache requires capacity");
    if (config->scoreScale <= 0.0f || !std::isfinite(config->scoreScale))
        throw std::invalid_argument("DIFFUSION_ATTENTION score scale must be finite and positive");

    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.x, config->nQueryHeads * config->headDim);
    ASSERT_EQ(context->outputSize.x, context->inputSize.x);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);

    const NnSize3D *keySize = &context->bufferConfigs[config->currentKeyBufferIndex].size;
    const NnSize3D *valueSize = &context->bufferConfigs[config->currentValueBufferIndex].size;
    ASSERT_EQ(keySize->floatType, F_32);
    ASSERT_EQ(valueSize->floatType, F_32);
    ASSERT_EQ(keySize->y, context->nBatches);
    ASSERT_EQ(valueSize->y, context->nBatches);
    ASSERT_EQ(keySize->x, config->nKvHeads * config->headDim);
    ASSERT_EQ(valueSize->x, keySize->x);
}

static void diffusionAttentionForward_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnDiffusionAttentionOpConfig *config = (NnDiffusionAttentionOpConfig *)context->opConfig;
    const float *positions = (const float *)context->pipes[config->positionPipeIndex];
    const float prefixLengthValue = *(const float *)context->pipes[config->prefixLengthPipeIndex];
    if (!std::isfinite(prefixLengthValue) || prefixLengthValue < 0.0f ||
        std::floor(prefixLengthValue) != prefixLengthValue)
        throw std::runtime_error("DIFFUSION_ATTENTION prefix length must be a non-negative integer");
    for (NnUint i = 0u; i < batchSize; i++) {
        if (!std::isfinite(positions[i]) || positions[i] < 0.0f ||
            std::floor(positions[i]) != positions[i])
            throw std::runtime_error("DIFFUSION_ATTENTION positions must be non-negative integers");
    }

    diffusionAttention_F32(
        (float *)context->output[0],
        (const float *)context->input[0],
        (const float *)context->buffers[config->currentKeyBufferIndex],
        (const float *)context->buffers[config->currentValueBufferIndex],
        (const float *)context->buffers[config->keyCacheBufferIndex],
        (const float *)context->buffers[config->valueCacheBufferIndex],
        positions,
        (NnUint)prefixLengthValue,
        batchSize,
        config,
        nThreads,
        threadIndex);
}

static void initTiedEmbeddingPartialForward(NnCpuOpContext *context) {
    const NnTiedEmbeddingPartialOpConfig *config =
        (NnTiedEmbeddingPartialOpConfig *)context->opConfig;
    if (context->weightSize.floatType != F_32 && context->weightSize.floatType != F_Q40)
        throw std::invalid_argument("TIED_EMBEDDING_PARTIAL supports F32 and Q40 embeddings");
    if (context->inputSize.floatType != F_32 && context->inputSize.floatType != F_Q80)
        throw std::invalid_argument("TIED_EMBEDDING_PARTIAL supports F32 and Q80 hidden states");
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.x, context->weightSize.x);
    ASSERT_EQ(context->outputSize.x, context->weightSize.y);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    assert(config->tileSize > 0u && config->tileCount > 0u);
    assert(config->tileCount == (context->weightSize.y + config->tileSize - 1u) / config->tileSize);
    assert(config->physicalSize >= context->inputSize.x + 2u);
    assert(config->logitSoftcap > 0.0f && std::isfinite(config->logitSoftcap));
    const NnBufferConfig *partial = &context->bufferConfigs[config->partialBufferIndex];
    ASSERT_EQ(partial->size.floatType, F_32);
    ASSERT_EQ(partial->size.x, config->tileCount * config->physicalSize);
    ASSERT_EQ(partial->size.y, context->nBatches);
    const NnPipeConfig *temperature = &context->pipeConfigs[config->temperaturePipeIndex];
    ASSERT_EQ(temperature->size.floatType, F_32);
    assert(temperature->size.length == 1u);
}

static void initTiedEmbeddingReduceForward(NnCpuOpContext *context) {
    const NnTiedEmbeddingReduceOpConfig *config =
        (NnTiedEmbeddingReduceOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.x, config->tileCount * config->physicalSize);
    ASSERT_EQ(context->outputSize.x, 1u);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    assert(context->weightSize.nBytes == 0u);
    assert(config->physicalSize >= config->hiddenSize + 2u && config->tileCount > 0u);
    const NnBufferConfig *mix = &context->bufferConfigs[config->embeddingMixBufferIndex];
    ASSERT_EQ(mix->size.floatType, F_32);
    ASSERT_EQ(mix->size.x, config->physicalSize);
    ASSERT_EQ(mix->size.y, context->nBatches);
}

static void initReduceMaxForward(NnCpuOpContext *context) {
    const NnReduceMaxOpConfig *config = (NnReduceMaxOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.x, 1u);
    assert(context->weightSize.nBytes == 0u);
    if (config->useDivisor != 0u) {
        ASSERT_EQ(context->pipeConfigs[config->divisorPipeIndex].size.x, 1u);
        ASSERT_EQ(context->pipeConfigs[config->divisorPipeIndex].size.y, 1u);
    }
}

static void initEmbeddingMixRescaleForward(NnCpuOpContext *context) {
    const NnEmbeddingMixRescaleOpConfig *config =
        (NnEmbeddingMixRescaleOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    assert(context->weightSize.nBytes == 0u);
    const NnBufferConfig *maximum = &context->bufferConfigs[config->globalMaxBufferIndex];
    ASSERT_EQ(maximum->size.floatType, F_32);
    ASSERT_EQ(maximum->size.x, 1u);
    ASSERT_EQ(maximum->size.y, context->nBatches);
    const NnPipeConfig *localMaximum = &context->pipeConfigs[config->vocabularyMaxPipeIndex];
    ASSERT_EQ(localMaximum->size.floatType, F_32);
    ASSERT_EQ(localMaximum->size.x, config->nNodes);
    ASSERT_EQ(localMaximum->size.y, context->nBatches);
    assert(config->nNodes > 0u && config->nodeIndex < config->nNodes);
}

static void initEmbeddingMixFinalizeForward(NnCpuOpContext *context) {
    const NnEmbeddingMixFinalizeOpConfig *config =
        (NnEmbeddingMixFinalizeOpConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    assert(context->weightSize.nBytes == 0u);
    const NnUint physicalSize = config->physicalSize == 0u
        ? context->outputSize.x + 1u
        : config->physicalSize;
    ASSERT_EQ(context->inputSize.x, physicalSize);
    assert(physicalSize >= context->outputSize.x + 1u);
}

static void tiedEmbeddingPartialForward_F32_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnTiedEmbeddingPartialOpConfig *config =
        (NnTiedEmbeddingPartialOpConfig *)context->opConfig;
    const float temperature = *(const float *)context->pipes[config->temperaturePipeIndex];
    if (!(temperature > 0.0f) || !std::isfinite(temperature))
        throw std::runtime_error("TIED_EMBEDDING temperature must be finite and positive");
    SPLIT_THREADS(batchStart, batchEnd, batchSize, nThreads, threadIndex);
    tiedEmbeddingPartial_F32_F32_F32(
        context->output,
        context->input,
        reinterpret_cast<const float *>(context->weight),
        reinterpret_cast<float *>(context->buffers[config->partialBufferIndex]),
        context->weightSize.y,
        context->weightSize.x,
        config->physicalSize,
        config->tileCount,
        1.0f / temperature,
        config->logitSoftcap,
        batchSize,
        batchStart,
        batchEnd);
}

static void tiedEmbeddingPartialForward_Q80_Q40_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnTiedEmbeddingPartialOpConfig *config =
        (NnTiedEmbeddingPartialOpConfig *)context->opConfig;
    const float temperature = *(const float *)context->pipes[config->temperaturePipeIndex];
    if (!(temperature > 0.0f) || !std::isfinite(temperature))
        throw std::runtime_error("TIED_EMBEDDING temperature must be finite and positive");
    SPLIT_THREADS(batchStart, batchEnd, batchSize, nThreads, threadIndex);
    tiedEmbeddingPartial_Q80_Q40_F32(
        context->output,
        context->input,
        reinterpret_cast<const NnBlockQ40 *>(context->weight),
        reinterpret_cast<float *>(context->buffers[config->partialBufferIndex]),
        context->weightSize.y,
        context->weightSize.x,
        config->physicalSize,
        config->tileCount,
        1.0f / temperature,
        config->logitSoftcap,
        batchStart,
        batchEnd);
}

static void tiedEmbeddingReduceForward_F32_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnTiedEmbeddingReduceOpConfig *config =
        (NnTiedEmbeddingReduceOpConfig *)context->opConfig;
    SPLIT_THREADS(batchStart, batchEnd, batchSize, nThreads, threadIndex);
    tiedEmbeddingReduce_F32(
        context->output,
        context->input,
        reinterpret_cast<float *>(context->buffers[config->embeddingMixBufferIndex]),
        config->hiddenSize,
        config->physicalSize,
        config->tileCount,
        batchStart,
        batchEnd);
}

static void reduceMaxForward_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnReduceMaxOpConfig *config = (NnReduceMaxOpConfig *)context->opConfig;
    float logitScale = 1.0f;
    if (config->useDivisor != 0u) {
        const float divisor = *(const float *)context->pipes[config->divisorPipeIndex];
        if (!(divisor > 0.0f) || !std::isfinite(divisor))
            throw std::runtime_error("REDUCE_MAX divisor must be finite and positive");
        logitScale = 1.0f / divisor;
    }
    for (NnUint batchIndex = 0u; batchIndex < batchSize; batchIndex++) {
        const float *input = (const float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        if (threadIndex == 0u)
            output[0] = *std::max_element(input, input + context->inputSize.x) * logitScale;
    }
}

static void embeddingMixRescaleForward_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnEmbeddingMixRescaleOpConfig *config =
        (NnEmbeddingMixRescaleOpConfig *)context->opConfig;
    const float *localMaximum = reinterpret_cast<const float *>(
        context->pipes[config->vocabularyMaxPipeIndex]);
    const float *globalMaximum = reinterpret_cast<const float *>(
        context->buffers[config->globalMaxBufferIndex]);
    for (NnUint batch = 0u; batch < batchSize; ++batch) {
        const float scale = expf(
            localMaximum[batch * config->nNodes + config->nodeIndex] - globalMaximum[batch]);
        const float *input = reinterpret_cast<const float *>(context->input[batch]);
        float *output = reinterpret_cast<float *>(context->output[batch]);
        SPLIT_THREADS(start, end, context->outputSize.x, nThreads, threadIndex);
        for (NnUint i = start; i < end; ++i)
            output[i] = input[i] * scale;
    }
}

static void embeddingMixFinalizeForward_F32_F32(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context)
{
    const NnEmbeddingMixFinalizeOpConfig *config =
        (NnEmbeddingMixFinalizeOpConfig *)context->opConfig;
    for (NnUint batch = 0u; batch < batchSize; ++batch) {
        const float *input = reinterpret_cast<const float *>(context->input[batch]);
        float *output = reinterpret_cast<float *>(context->output[batch]);
        const NnUint hiddenSize = context->outputSize.x;
        const float denominator = input[hiddenSize];
        if (!(denominator > 0.0f) || !std::isfinite(denominator))
            throw std::runtime_error("EMBEDDING_MIX_FINALIZE denominator must be finite and positive");
        const float factor = config->embeddingScale / denominator;
        SPLIT_THREADS(start, end, hiddenSize, nThreads, threadIndex);
        for (NnUint h = start; h < end; ++h)
            output[h] = input[h] * factor;
    }
}

static void initVocabStatsForward(NnCpuOpContext *context) {
    const NnVocabStatsOpConfig *config = (NnVocabStatsOpConfig *)context->opConfig;
    assert(context->weightSize.nBytes == 0u);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    if (config->mode == VOCAB_STATS_LOCAL) {
        ASSERT_EQ(context->outputSize.x, config->microshardCount * NN_VOCAB_STATS_WIDTH);
        if (config->microshardCount == 0u || context->inputSize.x % config->microshardCount != 0u)
            throw std::invalid_argument("VOCAB_STATS local vocabulary must divide into canonical microshards");
        ASSERT_EQ(context->bufferConfigs[config->globalMaxBufferIndex].size.x, 1u);
        ASSERT_EQ(context->pipeConfigs[config->temperaturePipeIndex].size.x, 1u);
        ASSERT_EQ(context->pipeConfigs[config->temperaturePipeIndex].size.y, 1u);
        if (config->useTargetTokens != 0u)
            ASSERT_EQ(context->pipeConfigs[config->targetTokensPipeIndex].size.x, 1u);
    } else if (config->mode == VOCAB_STATS_SAMPLE) {
        if (context->inputSize.x == 0u || context->inputSize.x % NN_VOCAB_STATS_WIDTH != 0u)
            throw std::invalid_argument("VOCAB_STATS sample input must contain complete node records");
        ASSERT_EQ(context->outputSize.x, 1u);
        ASSERT_EQ(context->pipeConfigs[config->temperaturePipeIndex].size.x, 1u);
        ASSERT_EQ(context->pipeConfigs[config->temperaturePipeIndex].size.y, 1u);
        ASSERT_EQ(context->pipeConfigs[config->samplingCoinsPipeIndex].size.x, 1u);
        if (config->microshardCount == 0u ||
            config->microshardStart + config->microshardCount > context->inputSize.x / NN_VOCAB_STATS_WIDTH)
            throw std::invalid_argument("VOCAB_STATS microshard range is out of bounds");
    } else {
        throw std::invalid_argument("VOCAB_STATS has an invalid mode");
    }
}

static double shiftedVocabLogit(float logit, float inverseTemperature, float scaledMaximum) {
    // Match the F32 multiplication used by tied-embedding maxima before
    // widening for accumulation. Double division would give the maximum a
    // nonzero shifted score and can make near-zero entropy negative.
    const float scaledLogit = logit * inverseTemperature;
    return static_cast<double>(scaledLogit) - scaledMaximum;
}

static void vocabStatsForward(
    NnUint nThreads,
    NnUint threadIndex,
    NnUint batchSize,
    NnCpuOpContext *context) {
    if (threadIndex != 0u)
        return;
    const NnVocabStatsOpConfig *config = (NnVocabStatsOpConfig *)context->opConfig;
    const float temperature = *(const float *)context->pipes[config->temperaturePipeIndex];
    if (!(temperature > 0.0f) || !std::isfinite(temperature))
        throw std::runtime_error("VOCAB_STATS temperature must be finite and positive");
    const float inverseTemperature = 1.0f / temperature;
    for (NnUint batchIndex = 0u; batchIndex < batchSize; ++batchIndex) {
        const float *input = (const float *)context->input[batchIndex];
        float *output = (float *)context->output[batchIndex];
        if (config->mode == VOCAB_STATS_LOCAL) {
            const float globalMax = ((const float *)context->buffers[config->globalMaxBufferIndex])[batchIndex];
            const NnUint microshardSize = context->inputSize.x / config->microshardCount;
            for (NnUint localShard = 0u; localShard < config->microshardCount; ++localShard) {
                double denominator = 0.0;
                double moment = 0.0;
                float localMax = -INFINITY;
                NnUint argmax = 0u;
                const NnUint start = localShard * microshardSize;
                const NnUint end = start + microshardSize;
                for (NnUint token = start; token < end; ++token) {
                    const float logit = input[token];
                    if (!std::isfinite(logit))
                        throw std::runtime_error("VOCAB_STATS logits must be finite");
                    if (logit > localMax) {
                        localMax = logit;
                        argmax = token;
                    }
                    const double shifted = shiftedVocabLogit(logit, inverseTemperature, globalMax);
                    const double weight = std::exp(shifted);
                    denominator += weight;
                    moment += weight * shifted;
                }
                float *record = output + localShard * NN_VOCAB_STATS_WIDTH;
                record[0] = (float)denominator;
                record[1] = (float)moment;
                record[2] = localMax;
                record[3] = (float)(config->vocabStart + argmax);
                record[4] = -INFINITY;
                if (config->useTargetTokens != 0u) {
                    const float targetValue = ((const float *)context->pipes[config->targetTokensPipeIndex])[batchIndex];
                    if (!std::isfinite(targetValue) || targetValue < 0.0f || std::floor(targetValue) != targetValue)
                        throw std::runtime_error("VOCAB_STATS target token must be a non-negative integer");
                    const NnUint target = (NnUint)targetValue;
                    const NnUint globalStart = config->vocabStart + start;
                    const NnUint globalEnd = config->vocabStart + end;
                    if (target >= globalStart && target < globalEnd)
                        record[4] = input[target - config->vocabStart];
                }
            }
        } else {
            const NnUint nMicroshards = context->inputSize.x / NN_VOCAB_STATS_WIDTH;
            double denominator = 0.0;
            for (NnUint shard = 0u; shard < nMicroshards; ++shard) {
                const float shardDenominator = input[shard * NN_VOCAB_STATS_WIDTH];
                if (!(shardDenominator > 0.0f) || !std::isfinite(shardDenominator))
                    throw std::runtime_error("VOCAB_STATS shard denominator must be finite and positive");
                denominator += shardDenominator;
            }
            const float coin = ((const float *)context->pipes[config->samplingCoinsPipeIndex])[batchIndex];
            if (!(coin >= 0.0f && coin < 1.0f))
                throw std::runtime_error("VOCAB_STATS sampling coin must be in [0, 1)");
            const double target = coin * denominator;
            double preceding = 0.0;
            NnUint owner = nMicroshards - 1u;
            for (NnUint shard = 0u; shard < nMicroshards; ++shard) {
                const double next = preceding + input[shard * NN_VOCAB_STATS_WIDTH];
                if (target < next) {
                    owner = shard;
                    break;
                }
                preceding = next;
            }
            output[0] = -1.0f;
            if (owner < config->microshardStart ||
                owner >= config->microshardStart + config->microshardCount)
                continue;
            const NnSize3D &logitsSize = context->pipeConfigs[config->localLogitsPipeIndex].size;
            const float *logits = (const float *)context->pipes[config->localLogitsPipeIndex] +
                (NnSize)batchIndex * logitsSize.x;
            const float globalMax = ((const float *)context->buffers[config->globalMaxBufferIndex])[batchIndex];
            const double localTarget = target - preceding;
            double cumulative = 0.0;
            const NnUint microshardSize = logitsSize.x / config->microshardCount;
            const NnUint localShard = owner - config->microshardStart;
            const NnUint start = localShard * microshardSize;
            const NnUint end = start + microshardSize;
            NnUint sampled = end - 1u;
            for (NnUint token = start; token < end; ++token) {
                cumulative += std::exp(shiftedVocabLogit(logits[token], inverseTemperature, globalMax));
                if (localTarget < cumulative) {
                    sampled = token;
                    break;
                }
            }
            output[0] = (float)(config->vocabStart + sampled);
        }
    }
}

static void initMulForward(NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
    if (config->useWeight != 0u) {
        ASSERT_EQ(context->weightSize.floatType, F_32);
        assert(context->weightSize.length == context->outputSize.x);
    } else {
        assert(context->weightSize.nBytes == 0u);
    }
}

static void mulForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMulOpCodeConfig *config = (NnMulOpCodeConfig *)context->opConfig;
    const float *multiplier = config->useWeight != 0u
        ? (const float *)context->weight
        : (const float *)context->buffers[config->multiplierBufferIndex];

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            mul_F32(
                (float *)context->output[zOffset + y],
                (float *)context->input[zOffset + y],
                config->useWeight != 0u ? multiplier : &multiplier[context->outputSize.x * (zOffset + y)],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void scaleForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnScaleOpCodeConfig *config = (NnScaleOpCodeConfig *)context->opConfig;
    const float *scale = config->useScalar == 0u ? (const float *)context->buffers[config->scaleBufferIndex] : nullptr;

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            const NnUint index = z * context->inputSize.y + y;
            const float s = config->useScalar != 0u ? config->scalar : scale[index];
            const float *i = (float *)context->input[index];
            float *o = (float *)context->output[index];
            scale_F32(i, o, s, context->inputSize.x, nThreads, threadIndex);
        }
    }
}

static void initCastForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, context->outputSize.z);
}

static void castForward_ANY(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnUint rowBytes = getBytes(context->outputSize.floatType, context->outputSize.x);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            copy_UNK(
                context->output[zOffset + y],
                context->input[zOffset + y],
                rowBytes,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            quantizeF32toQ80(
                (float *)context->input[zOffset + y],
                (NnBlockQ80 *)context->output[zOffset + y],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void castForward_Q80_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_Q80);
    ASSERT_EQ(context->outputSize.floatType, F_32);

    for (NnUint z = 0u; z < context->inputSize.z; z++) {
        const NnUint zOffset = z * context->inputSize.y;
        for (NnUint y = 0u; y < batchSize; y++) {
            dequantizeQ80toF32(
                (NnBlockQ80 *)context->input[zOffset + y],
                (float *)context->output[zOffset + y],
                context->outputSize.x,
                nThreads,
                threadIndex);
        }
    }
}

static void initRepeatZForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.x, context->outputSize.x);
    ASSERT_EQ(context->inputSize.y, context->outputSize.y);
    ASSERT_EQ(context->inputSize.z, 1u);
    assert(context->inputSize.z <= context->outputSize.z);
    assert(context->inputSize.z > 0u);
}

static void repeatZForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    const NnSize rowBytes = getBytes(F_32, context->outputSize.x);
    for (NnUint z = 0u; z < context->outputSize.z; ++z) {
        for (NnUint y = 0u; y < batchSize; ++y)
            copy_UNK(context->output[z * context->outputSize.y + y], context->input[y], rowBytes, nThreads, threadIndex);
    }
}

static void repeatZForward_F32_Q80(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_Q80);
    const NnSize dimSize = getBytes(F_Q80, context->outputSize.x);

    for (NnUint z = 0u; z < context->outputSize.z; z++) {
        for (NnUint y = 0u; y < batchSize; y++) {
            NnByte *output = context->output[z * context->outputSize.y + y];
            if (z == 0u) {
                quantizeF32toQ80(
                    (float *)context->input[y],
                    (NnBlockQ80 *)output,
                    context->outputSize.x,
                    nThreads,
                    threadIndex);
            } else {
                copy_UNK(
                    output,
                    context->output[y],
                    dimSize,
                    nThreads,
                    threadIndex);
            }
        }
    }
}

NnUint resolveShiftSlot(float absolutePosition, NnUint capacity, NnUint nSlots) {
    if (!std::isfinite(absolutePosition) || absolutePosition < 0.0f || std::floor(absolutePosition) != absolutePosition)
        throw std::runtime_error("SHIFT position must be a finite non-negative integer");
    if (absolutePosition > (float)UINT32_MAX)
        throw std::runtime_error("SHIFT position is too large");
    if (nSlots == 0u)
        throw std::runtime_error("SHIFT output has no slots");
    if (capacity > nSlots)
        throw std::runtime_error("SHIFT capacity exceeds output slots");

    const NnUint position = (NnUint)absolutePosition;
    const NnUint slot = capacity == 0u ? position : position % capacity;
    if (slot >= nSlots)
        throw std::runtime_error("SHIFT slot is outside the output");
    return slot;
}

void validateShiftIndexes(const float *absolutePositions, NnUint count, NnUint capacity, NnUint nSlots) {
    for (NnUint i = 0u; i < count; i++) {
        const NnUint slot = resolveShiftSlot(absolutePositions[i], capacity, nSlots);
        for (NnUint j = 0u; j < i; j++) {
            if (slot == resolveShiftSlot(absolutePositions[j], capacity, nSlots))
                throw std::runtime_error("SHIFT invocation contains duplicate cache slots");
        }
    }
}

static void initShiftForward(NnCpuOpContext *context) {
    ASSERT_EQ(context->hasInputContinuousMemory, true);
    ASSERT_EQ(context->hasOutputContinuousMemory, true);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->outputSize.y, 1u);
    assert(context->inputSize.x > 0u);
    assert(context->outputSize.x % context->inputSize.x == 0u);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const NnUint nSlots = context->outputSize.x / context->inputSize.x;
    if (config->capacity > nSlots)
        throw std::runtime_error("SHIFT capacity exceeds output slots");
}

static void shiftForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    ASSERT_EQ(context->hasInputContinuousMemory, true);
    ASSERT_EQ(context->hasOutputContinuousMemory, true);
    ASSERT_EQ(context->inputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.floatType, F_32);
    ASSERT_EQ(context->outputSize.y, 1);

    const NnShiftOpCodeConfig *config = (NnShiftOpCodeConfig *)context->opConfig;
    const float *indexes = (float *)context->pipes[config->indexPipeIndex];
    const NnSize dimBytes = getBytes(F_32, context->inputSize.x);
    const NnUint nSlots = context->outputSize.x / context->inputSize.x;
    NnByte *output = context->output[0];

    validateShiftIndexes(indexes, batchSize, config->capacity, nSlots);

    for (NnUint batchIndex = 0; batchIndex < batchSize; batchIndex++) {
        const NnSize index = resolveShiftSlot(indexes[batchIndex], config->capacity, nSlots);
        copy_UNK(
            &output[index * dimBytes],
            context->input[batchIndex],
            dimBytes,
            nThreads,
            threadIndex);
    }
}

static void softmaxForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    assert(*context->input == *context->output);

    for (NnUint y = threadIndex; y < batchSize; y += nThreads)
        softmax_F32(
            (float *)context->output[y],
            context->outputSize.x);
}

static void initMoeGateForward(NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    ASSERT_EQ(context->inputSize.z, 1u);
    ASSERT_EQ(context->inputSize.y, context->nBatches);
    assert(context->inputSize.x >= config->k);
    ASSERT_EQ(context->outputSize.z, config->k);
    ASSERT_EQ(context->outputSize.y, context->nBatches);
    ASSERT_EQ(context->outputSize.x, 1u);
    if (config->hasExpertScale != 0u) {
        if (config->expertScaleFromWeight != 0u) {
            ASSERT_EQ(context->weightSize.floatType, F_32);
            assert(context->weightSize.length >= context->inputSize.x);
        } else {
            const NnSize3D *scaleSize = &context->bufferConfigs[config->expertScaleBufferIndex].size;
            ASSERT_EQ(scaleSize->floatType, F_32);
            assert(scaleSize->length >= context->inputSize.x);
        }
    }
}

static void moeGateForward_F32_F32(NnUint nThreads, NnUint threadIndex, NnUint batchSize, NnCpuOpContext *context) {
    const NnMoeGateOpCodeConfig *config = (NnMoeGateOpCodeConfig *)context->opConfig;
    float *indexes = (float *)context->buffers[config->indexesBufferIndex];
    const float *expertScale = config->hasExpertScale == 0u ? nullptr
        : config->expertScaleFromWeight != 0u ? (const float *)context->weight
        : (const float *)context->buffers[config->expertScaleBufferIndex];

    std::vector<NnUint> pos(config->k);
    std::vector<float> weights(config->k);
    for (NnUint y = threadIndex; y < batchSize; y += nThreads) {
        float *input = (float *)context->input[y];
        routeExperts_F32(input, context->inputSize.x, config->k, config->normTopk == 1u,
            expertScale, pos.data(), weights.data());

        for (NnUint k = 0u; k < config->k; k++) {
            const NnUint p = pos[k];
            indexes[y * config->k + k] = (float)p;

            // (nActiveExperts, nBatches, 1)
            float *output = (float *)context->output[k * context->outputSize.y + y];
            *output = weights[k];
        }

        DEBUG_VECTOR(context, "indexes", (&indexes[y * config->k]));
    }
}

// device

void printCpuInstructionSet() {
    printf("🧠 CPU:");
#if defined(__ARM_NEON)
    printf(" neon");
#if defined(__ARM_FEATURE_DOTPROD)
    printf(" dotprod");
#endif
#if defined(__ARM_FP16_FORMAT_IEEE)
    printf(" fp16");
#endif
#endif
#if defined(__AVX2__)
    printf(" avx2");
#endif
#if defined(__AVX512F__)
    printf(" avx512f");
#endif
    printf("\n");
}

NnCpuOpForwardInit getCpuOpForwardInit(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_EMBEDDING)
        return initEmbeddingForward;
    if (code == OP_INV_RMS)
        return initInvRmsForward;
    if (code == OP_RMS_NORM)
        return initRmsNormForward_ANY_F32_F32;
    if (code == OP_ROPE)
        return initRopeForward_F32;
    if (code == OP_MULTIHEAD_ATT)
        return initMultiHeadAttForward;
    if (code == OP_MATMUL)
        return initMatmulForward;
    if (code == OP_MUL)
        return initMulForward;
    if (code == OP_CAST)
        return initCastForward;
    if (code == OP_REPEAT_Z)
        return initRepeatZForward;
    if (code == OP_SHIFT)
        return initShiftForward;
    if (code == OP_MOE_GATE)
        return initMoeGateForward;
    if (code == OP_DIFFUSION_ATTENTION)
        return initDiffusionAttentionForward;
    if (code == OP_TIED_EMBEDDING_PARTIAL)
        return initTiedEmbeddingPartialForward;
    if (code == OP_TIED_EMBEDDING_REDUCE)
        return initTiedEmbeddingReduceForward;
    if (code == OP_REDUCE_MAX)
        return initReduceMaxForward;
    if (code == OP_EMBEDDING_MIX_RESCALE)
        return initEmbeddingMixRescaleForward;
    if (code == OP_EMBEDDING_MIX_FINALIZE)
        return initEmbeddingMixFinalizeForward;
    if (code == OP_VOCAB_STATS)
        return initVocabStatsForward;
    return nullptr;
}

NnCpuOpForward getCpuOpForward(NnOpCode code, NnOpQuantType quantType) {
    if (code == OP_MERGE_ADD) {
        if (quantType == F32_F32_F32) return mergeAddForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mergeAddForward_Q80_F32;
    }
    if (code == OP_MERGE_SUM) {
        if (quantType == F32_F32_F32) return mergeSumForward_F32_F32;
        if (quantType == Q80_Q80_F32) return mergeSumForward_Q80_F32;
    }
    if (code == OP_EMBEDDING) {
        if (quantType == F32_F32_F32) return embeddingForward_F32_F32_F32;
        if (quantType == F32_Q40_F32) return embeddingForward_F32_Q40_F32;
        if (quantType == F32_F32_Q80) return embeddingForward_F32_F32_Q80;
    }
    if (code == OP_INV_RMS) {
        if (quantType == F32_F32_F32) return invRmsForward_F32_F32;
    }
    if (code == OP_RMS_NORM) {
        if (quantType == F32_F32_F32) return rmsNormForward_F32_F32_F32;
        if (quantType == Q80_F32_F32) return rmsNormForward_Q80_F32_F32;
    }
    if (code == OP_MATMUL) {
        if (quantType == F32_F32_F32) return matmulForward_F32_F32_F32;
        if (quantType == Q80_Q40_F32) return matmulForward_Q80_Q40_F32;
    }
    if (code == OP_ROPE) {
        if (quantType == F32_F32_F32) return ropeForward_F32_F32;
    }
    if (code == OP_MULTIHEAD_ATT) {
        if (quantType == F32_F32_F32) return multiHeadAttForward_F32_F32;
    }
    if (code == OP_GELU) {
        if (quantType == F32_F32_F32) return geluForward_F32_F32_F32;
    }
    if (code == OP_SILU) {
        if (quantType == F32_F32_F32) return siluForward_F32_F32;
    }
    if (code == OP_MUL) {
        if (quantType == F32_F32_F32) return mulForward_F32_F32;
    }
    if (code == OP_SCALE) {
        if (quantType == F32_F32_F32) return scaleForward_F32_F32;
    }
    if (code == OP_CAST) {
        if (quantType == F32_F32_F32) return castForward_ANY;
        if (quantType == F32_F32_Q80) return castForward_F32_Q80;
        if (quantType == Q80_Q80_Q80) return castForward_ANY;
        if (quantType == Q80_Q80_F32) return castForward_Q80_F32;
    }
    if (code == OP_REPEAT_Z) {
        if (quantType == F32_F32_F32) return repeatZForward_F32_F32;
        if (quantType == F32_F32_Q80) return repeatZForward_F32_Q80;
    }
    if (code == OP_SHIFT) {
        if (quantType == F32_F32_F32) return shiftForward_F32_F32;
    }
    if (code == OP_SOFTMAX) {
        if (quantType == F32_F32_F32) return softmaxForward_F32_F32;
    }
    if (code == OP_MOE_GATE) {
        if (quantType == F32_F32_F32) return moeGateForward_F32_F32;
    }
    if (code == OP_DIFFUSION_ATTENTION) {
        if (quantType == F32_F32_F32) return diffusionAttentionForward_F32_F32;
    }
    if (code == OP_TIED_EMBEDDING_PARTIAL) {
        if (quantType == F32_F32_F32) return tiedEmbeddingPartialForward_F32_F32_F32;
        if (quantType == Q80_Q40_F32) return tiedEmbeddingPartialForward_Q80_Q40_F32;
    }
    if (code == OP_TIED_EMBEDDING_REDUCE) {
        if (quantType == F32_F32_F32) return tiedEmbeddingReduceForward_F32_F32_F32;
    }
    if (code == OP_REDUCE_MAX) {
        if (quantType == F32_F32_F32) return reduceMaxForward_F32_F32;
    }
    if (code == OP_EMBEDDING_MIX_RESCALE) {
        if (quantType == F32_F32_F32) return embeddingMixRescaleForward_F32_F32;
    }
    if (code == OP_EMBEDDING_MIX_FINALIZE) {
        if (quantType == F32_F32_F32) return embeddingMixFinalizeForward_F32_F32;
    }
    if (code == OP_VOCAB_STATS) {
        if (quantType == F32_F32_F32) return vocabStatsForward;
    }
    if (code == OP_TANH) {
        if (quantType == F32_F32_F32) return tanhForward_F32_F32;
    }
    return nullptr;
}
