/**
 * AuraScale Kernel — Catmull-Rom (Mitchell-Netravali B=0, C=0.5) Upscaler
 *
 * Single-pass spatial interpolation with intrinsic edge-sharpening via
 * negative lobes in the BC-Spline kernel.
 *
 * Optimization Stack:
 *   - AVX2 + FMA intrinsics: 8-wide float vectorization
 *   - 256x256 tiling:        Optimized for 1MB L1 cache residency
 *   - Row-major stride-1:    Spatial locality for sequential access
 *   - Loop interchange:      (y, x) ordering
 *   - Loop unrolling x4:     Saturates instruction pipeline
 *   - std::jthread stripping: Horizontal parallelism across all cores
 */

#include "aurascale_kernel.hpp"

#include <immintrin.h>
#include <thread>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cstdlib>

// ============================================================================
// Mitchell-Netravali BC-Spline Framework (B=0, C=0.5 => Catmull-Rom)
//
// For |x| <= 1:  1.5|x|^3 - 2.5|x|^2 + 1
// For |x| <  2: -0.5|x|^3 + 2.5|x|^2 - 4|x| + 2
// Otherwise:      0
//
// The negative lobes (at 1 < |x| < 2) provide intrinsic high-pass
// enhancement, sharpening edges during the interpolation itself.
// ============================================================================

// --- AVX2 vectorized weight calculation for 8 pixels simultaneously ---------
inline __m256 catmullrom_weights_avx2(__m256 x) {
    __m256 abs_x = _mm256_andnot_ps(_mm256_set1_ps(-0.0f), x);
    __m256 x2    = _mm256_mul_ps(abs_x, abs_x);
    __m256 x3    = _mm256_mul_ps(x2, abs_x);

    // Region 1: |x| <= 1  =>  1.5*x3 - 2.5*x2 + 1.0
    __m256 mask1 = _mm256_cmp_ps(abs_x, _mm256_set1_ps(1.0f), _CMP_LE_OQ);
    __m256 w1    = _mm256_fmadd_ps(
                       _mm256_set1_ps(1.5f), x3,
                       _mm256_fmadd_ps(
                           _mm256_set1_ps(-2.5f), x2,
                           _mm256_set1_ps(1.0f)));

    // Region 2: 1 < |x| < 2  =>  -0.5*x3 + 2.5*x2 - 4.0*|x| + 2.0
    __m256 mask2 = _mm256_andnot_ps(
                       mask1,
                       _mm256_cmp_ps(abs_x, _mm256_set1_ps(2.0f), _CMP_LT_OQ));
    __m256 w2    = _mm256_fmadd_ps(
                       _mm256_set1_ps(-0.5f), x3,
                       _mm256_fmadd_ps(
                           _mm256_set1_ps(2.5f), x2,
                           _mm256_fmadd_ps(
                               _mm256_set1_ps(-4.0f), abs_x,
                               _mm256_set1_ps(2.0f))));

    // Blend: region1 | region2 | 0
    __m256 result = _mm256_blendv_ps(_mm256_setzero_ps(), w2, mask2);
    result        = _mm256_blendv_ps(result, w1, mask1);

    return result;
}

// --- Scalar weight for Y-axis (shared across 8 vectorized X-pixels) ---------
inline float catmullrom_weight_scalar(float x) {
    float ax  = std::abs(x);
    float ax2 = ax * ax;
    float ax3 = ax2 * ax;
    if (ax <= 1.0f)
        return  1.5f * ax3 - 2.5f * ax2 + 1.0f;
    if (ax < 2.0f)
        return -0.5f * ax3 + 2.5f * ax2 - 4.0f * ax + 2.0f;
    return 0.0f;
}

// ============================================================================
// Naive baseline — scalar, single-threaded, no SIMD, no tiling.
// Used exclusively for benchmarking the speedup factor.
// ============================================================================
extern "C" {

AURASCALE_API void upscale_catmullrom_naive(
    const float* in_data, int in_w, int in_h, int channels,
    float* out_data, int out_w, int out_h)
{
    const float scale_x = static_cast<float>(in_w) / out_w;
    const float scale_y = static_cast<float>(in_h) / out_h;

    for (int c = 0; c < channels; ++c) {
        const float* ch_in  = in_data  + c * in_w  * in_h;
        float*       ch_out = out_data + c * out_w * out_h;

        // Loop interchange: y then x (row-major access)
        for (int y = 0; y < out_h; ++y) {
            const float v  = y * scale_y;
            const int   iy = static_cast<int>(std::floor(v));
            const float dy = v - iy;

            for (int x = 0; x < out_w; ++x) {
                const float u  = x * scale_x;
                const int   ix = static_cast<int>(std::floor(u));
                const float dx = u - ix;

                float sum = 0.0f;
                for (int m = -1; m <= 2; ++m) {
                    const float wy = catmullrom_weight_scalar(dy - m);
                    const int   sy = std::clamp(iy + m, 0, in_h - 1);
                    for (int n = -1; n <= 2; ++n) {
                        const float wx = catmullrom_weight_scalar(dx - n);
                        const int   sx = std::clamp(ix + n, 0, in_w - 1);
                        sum += ch_in[sy * in_w + sx] * wy * wx;
                    }
                }
                ch_out[y * out_w + x] = sum;
            }
        }
    }
}

// ============================================================================
// Optimized engine — AVX2 + FMA + Tiling + Unrolling + jthread stripping
// ============================================================================
AURASCALE_API void upscale_catmullrom_optimized(
    const float* in_data, int in_w, int in_h, int channels,
    float* out_data, int out_w, int out_h)
{
    const float scale_x = static_cast<float>(in_w) / out_w;
    const float scale_y = static_cast<float>(in_h) / out_h;

    int num_threads = static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads == 0) num_threads = 4;

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    const int strip_height = out_h / num_threads;

    for (int t = 0; t < num_threads; ++t) {
        const int start_y = t * strip_height;
        const int end_y   = (t == num_threads - 1) ? out_h
                                                    : start_y + strip_height;

        threads.emplace_back([=]() {
            for (int c = 0; c < channels; ++c) {
                const float* ch_in  = in_data  + c * in_w  * in_h;
                float*       ch_out = out_data + c * out_w * out_h;

                // ---- Tiling (Blocking): 256x256 output blocks ----
                for (int ty = start_y; ty < end_y; ty += 256) {
                    for (int tx = 0; tx < out_w; tx += 256) {
                        const int block_end_y = std::min(ty + 256, end_y);
                        const int block_end_x = std::min(tx + 256, out_w);

                        // Loop interchange: y then x
                        for (int y = ty; y < block_end_y; ++y) {
                            const float v  = y * scale_y;
                            const int   iy = static_cast<int>(std::floor(v));
                            const float dy = v - iy;

                            // Precompute Y weights (shared across all X in row)
                            const float wy_m1 = catmullrom_weight_scalar(dy + 1.0f);
                            const float wy_0  = catmullrom_weight_scalar(dy);
                            const float wy_1  = catmullrom_weight_scalar(dy - 1.0f);
                            const float wy_2  = catmullrom_weight_scalar(dy - 2.0f);

                            // Precompute clamped Y row offsets (stride-1 prep)
                            const int sy_m1 = std::clamp(iy - 1, 0, in_h - 1) * in_w;
                            const int sy_0  = std::clamp(iy,     0, in_h - 1) * in_w;
                            const int sy_1  = std::clamp(iy + 1, 0, in_h - 1) * in_w;
                            const int sy_2  = std::clamp(iy + 2, 0, in_h - 1) * in_w;

                            // AVX2 vectorized path: 8 pixels per iteration
                            const int x_limit = tx + ((block_end_x - tx) & ~7);

                            for (int x = tx; x < x_limit; x += 8) {
                                // Build 8 source X coordinates
                                const __m256 x_vec  = _mm256_add_ps(
                                    _mm256_set1_ps(static_cast<float>(x)),
                                    _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0));
                                const __m256 u_vec  = _mm256_mul_ps(x_vec,
                                                          _mm256_set1_ps(scale_x));
                                const __m256 ix_f   = _mm256_floor_ps(u_vec);
                                const __m256i ix_i  = _mm256_cvtps_epi32(ix_f);
                                const __m256 dx_vec = _mm256_sub_ps(u_vec, ix_f);

                                // Compute X weights for 8 pixels (SIMD)
                                const __m256 wx_m1 = catmullrom_weights_avx2(
                                    _mm256_add_ps(dx_vec, _mm256_set1_ps(1.0f)));
                                const __m256 wx_0  = catmullrom_weights_avx2(dx_vec);
                                const __m256 wx_1  = catmullrom_weights_avx2(
                                    _mm256_sub_ps(dx_vec, _mm256_set1_ps(1.0f)));
                                const __m256 wx_2  = catmullrom_weights_avx2(
                                    _mm256_sub_ps(dx_vec, _mm256_set1_ps(2.0f)));

                                // Clamp X indices
                                const __m256i zero_v = _mm256_setzero_si256();
                                const __m256i max_v  = _mm256_set1_epi32(in_w - 1);

                                const __m256i sx_m1 = _mm256_min_epi32(
                                    _mm256_max_epi32(
                                        _mm256_add_epi32(ix_i, _mm256_set1_epi32(-1)),
                                        zero_v), max_v);
                                const __m256i sx_0  = _mm256_min_epi32(
                                    _mm256_max_epi32(ix_i, zero_v), max_v);
                                const __m256i sx_1  = _mm256_min_epi32(
                                    _mm256_max_epi32(
                                        _mm256_add_epi32(ix_i, _mm256_set1_epi32(1)),
                                        zero_v), max_v);
                                const __m256i sx_2  = _mm256_min_epi32(
                                    _mm256_max_epi32(
                                        _mm256_add_epi32(ix_i, _mm256_set1_epi32(2)),
                                        zero_v), max_v);

                                __m256 sum = _mm256_setzero_ps();

                                // ---- Manual Loop Unrolling (factor 4) ----
                                // Row m = -1
                                {
                                    const __m256 wy = _mm256_set1_ps(wy_m1);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_m1),
                                        _mm256_i32gather_ps(ch_in + sy_m1, sx_m1, 4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_0),
                                        _mm256_i32gather_ps(ch_in + sy_m1, sx_0,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_1),
                                        _mm256_i32gather_ps(ch_in + sy_m1, sx_1,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_2),
                                        _mm256_i32gather_ps(ch_in + sy_m1, sx_2,  4), sum);
                                }
                                // Row m = 0
                                {
                                    const __m256 wy = _mm256_set1_ps(wy_0);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_m1),
                                        _mm256_i32gather_ps(ch_in + sy_0, sx_m1, 4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_0),
                                        _mm256_i32gather_ps(ch_in + sy_0, sx_0,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_1),
                                        _mm256_i32gather_ps(ch_in + sy_0, sx_1,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_2),
                                        _mm256_i32gather_ps(ch_in + sy_0, sx_2,  4), sum);
                                }
                                // Row m = 1
                                {
                                    const __m256 wy = _mm256_set1_ps(wy_1);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_m1),
                                        _mm256_i32gather_ps(ch_in + sy_1, sx_m1, 4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_0),
                                        _mm256_i32gather_ps(ch_in + sy_1, sx_0,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_1),
                                        _mm256_i32gather_ps(ch_in + sy_1, sx_1,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_2),
                                        _mm256_i32gather_ps(ch_in + sy_1, sx_2,  4), sum);
                                }
                                // Row m = 2
                                {
                                    const __m256 wy = _mm256_set1_ps(wy_2);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_m1),
                                        _mm256_i32gather_ps(ch_in + sy_2, sx_m1, 4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_0),
                                        _mm256_i32gather_ps(ch_in + sy_2, sx_0,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_1),
                                        _mm256_i32gather_ps(ch_in + sy_2, sx_1,  4), sum);
                                    sum = _mm256_fmadd_ps(
                                        _mm256_mul_ps(wy, wx_2),
                                        _mm256_i32gather_ps(ch_in + sy_2, sx_2,  4), sum);
                                }

                                // Store 8 results (stride-1 sequential write)
                                _mm256_storeu_ps(ch_out + y * out_w + x, sum);
                            }

                            // Scalar remainder for non-8-aligned block widths
                            for (int x = x_limit; x < block_end_x; ++x) {
                                const float u  = x * scale_x;
                                const int   ix = static_cast<int>(std::floor(u));
                                const float dx = u - ix;

                                float sum = 0.0f;
                                for (int m = -1; m <= 2; ++m) {
                                    const float wy = catmullrom_weight_scalar(dy - m);
                                    const int   sy = std::clamp(iy + m, 0, in_h - 1);
                                    for (int n = -1; n <= 2; ++n) {
                                        const float wx = catmullrom_weight_scalar(dx - n);
                                        const int   sx = std::clamp(ix + n, 0, in_w - 1);
                                        sum += ch_in[sy * in_w + sx] * wy * wx;
                                    }
                                }
                                ch_out[y * out_w + x] = sum;
                            }
                        }
                    }
                }
            }
        });
    }
    // std::jthread destructors auto-join here
}

} // extern "C"
