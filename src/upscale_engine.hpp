#ifndef UPSCALE_ENGINE_HPP
#define UPSCALE_ENGINE_HPP

#ifdef _WIN32
#define UPSCALE_API __declspec(dllexport)
#else
#define UPSCALE_API __attribute__((visibility("default")))
#endif

extern "C" {

/**
 * Naive (scalar, single-threaded) Catmull-Rom upscaler.
 * Used exclusively as a baseline for benchmarking.
 *
 * @param in_data   Pointer to planar (C, H, W) float32 input image.
 * @param in_w      Width of the input image.
 * @param in_h      Height of the input image.
 * @param channels  Number of color channels (e.g. 3 for RGB).
 * @param out_data  Pointer to planar (C, out_H, out_W) float32 output buffer.
 * @param out_w     Width of the output image.
 * @param out_h     Height of the output image.
 */
UPSCALE_API void upscale_catmullrom_naive(
    const float* in_data, int in_w, int in_h, int channels,
    float* out_data, int out_w, int out_h);

/**
 * Optimized Catmull-Rom upscaler.
 * Uses AVX2 SIMD, FMA, 32x32 tiling, loop unrolling, and std::jthread
 * horizontal stripping for maximum throughput.
 *
 * The Catmull-Rom kernel (Mitchell-Netravali B=0, C=0.5) includes negative
 * lobes that provide mathematical high-pass sharpening during interpolation,
 * eliminating the need for a separate sharpening pass.
 *
 * @param in_data   Pointer to 32-byte aligned planar (C, H, W) float32 input.
 * @param in_w      Width of the input image.
 * @param in_h      Height of the input image.
 * @param channels  Number of color channels (e.g. 3 for RGB).
 * @param out_data  Pointer to 32-byte aligned planar (C, out_H, out_W) output.
 * @param out_w     Width of the output image.
 * @param out_h     Height of the output image.
 */
UPSCALE_API void upscale_catmullrom_optimized(
    const float* in_data, int in_w, int in_h, int channels,
    float* out_data, int out_w, int out_h);

} // extern "C"

#endif // UPSCALE_ENGINE_HPP
