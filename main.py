"""
AuraScale — Python Automation & Benchmarking Suite
===================================================
Loads a sample image, runs the Catmull-Rom upscaler (Naive vs Optimized),
and reports execution time, speedup factor, and throughput in Megapixels/sec.

The Catmull-Rom kernel (Mitchell-Netravali B=0, C=0.5) provides intrinsic
edge-sharpening via negative lobes — no separate sharpening pass required.
"""

import ctypes
import numpy as np
import time
import cv2
import os
import argparse


# ---------------------------------------------------------------------------
# Library discovery
# ---------------------------------------------------------------------------
lib_ext  = '.dll' if os.name == 'nt' else '.so'
base_dir = os.path.dirname(os.path.abspath(__file__))

# Search both lib-prefixed (MinGW/Clang) and non-prefixed (MSVC) names
search_paths = [
    os.path.join(base_dir, 'build', f'libaurascale_kernel{lib_ext}'),
    os.path.join(base_dir, 'build', f'aurascale_kernel{lib_ext}'),
    os.path.join(base_dir, 'build', 'Release', f'aurascale_kernel{lib_ext}'),
    os.path.join(base_dir, f'libaurascale_kernel{lib_ext}'),
    os.path.join(base_dir, f'aurascale_kernel{lib_ext}'),
]

lib_path = None
for p in search_paths:
    if os.path.exists(p):
        lib_path = p
        break

if lib_path is None:
    print("ERROR: Could not find aurascale_kernel library.")
    print("Build with CMake first:")
    print("  mkdir build && cd build")
    print("  cmake .. -DCMAKE_BUILD_TYPE=Release -G \"MinGW Makefiles\"")
    print("  cmake --build . --config Release")
    exit(1)

print(f"[+] Loaded library: {lib_path}")
engine = ctypes.CDLL(lib_path)


# ---------------------------------------------------------------------------
# C function signatures (Planar float32: [C, H, W])
# ---------------------------------------------------------------------------
_planar_ptr = np.ctypeslib.ndpointer(dtype=np.float32, ndim=3, flags='C_CONTIGUOUS')

engine.upscale_catmullrom_naive.argtypes = [
    _planar_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _planar_ptr, ctypes.c_int, ctypes.c_int,
]
engine.upscale_catmullrom_naive.restype = None

engine.upscale_catmullrom_optimized.argtypes = [
    _planar_ptr, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    _planar_ptr, ctypes.c_int, ctypes.c_int,
]
engine.upscale_catmullrom_optimized.restype = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def generate_test_image(height: int, width: int) -> np.ndarray:
    """Generates a gradient + wave test pattern (single channel)."""
    x = np.linspace(0, 255, width)
    y = np.linspace(0, 255, height)
    xv, yv = np.meshgrid(x, y)
    img = (xv + yv) / 2.0 + np.sin(xv / 10.0) * 20.0 + np.cos(yv / 10.0) * 20.0
    return np.clip(img, 0, 255).astype(np.float32)


def to_hwc(img_chw: np.ndarray) -> np.ndarray:
    """Convert planar (C,H,W) float32 to interleaved (H,W,C) uint8."""
    return np.clip(np.transpose(img_chw, (1, 2, 0)), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
def benchmark(scale_factor: float = 30.0, input_path: str = None) -> None:
    # --- Load or generate input ---
    img_file = input_path or "test_image.jpg"
    if os.path.exists(img_file):
        print(f"[*] Loading {img_file} (RGB)...")
        img_bgr = cv2.imread(img_file, cv2.IMREAD_COLOR)
        if img_bgr is None:
            print(f"ERROR: Failed to decode '{img_file}'. Is it a valid image?")
            exit(1)
        in_h, in_w, in_c = img_bgr.shape
        # HWC -> CHW planar, contiguous float32
        in_img = np.ascontiguousarray(
            np.transpose(img_bgr, (2, 0, 1)).astype(np.float32))
    else:
        print("[*] Generating dummy 3-channel test image...")
        in_h, in_w, in_c = 256, 256, 3
        ch = generate_test_image(in_h, in_w)
        in_img = np.ascontiguousarray(
            np.stack([ch, ch, ch], axis=0).astype(np.float32))

    out_h = int(in_h * scale_factor)
    out_w = int(in_w * scale_factor)
    total_out_pixels = out_h * out_w * in_c

    print(f"{'=' * 60}")
    print(f"  AuraScale Benchmarking Suite")
    print(f"  Kernel : Catmull-Rom (Mitchell-Netravali B=0, C=0.5)")
    print(f"  Input  : {in_w} x {in_h}  ({in_c} channels)")
    print(f"  Output : {out_w} x {out_h}  (Scale: {scale_factor}x)")
    print(f"  Pixels : {total_out_pixels / 1e6:.2f} Megapixels")
    print(f"  Layout : Planar float32  (stride-1 row-major)")
    print(f"{'=' * 60}\n")

    # --- Allocate output buffers (Planar: CHW) ---
    out_naive = np.zeros((in_c, out_h, out_w), dtype=np.float32)
    out_opt   = np.zeros((in_c, out_h, out_w), dtype=np.float32)

    # --- Warmup (populate instruction cache, trigger JIT if any) ---
    print("[*] Warming up optimized engine...")
    engine.upscale_catmullrom_optimized(
        in_img, in_w, in_h, in_c, out_opt, out_w, out_h)

    # --- Naive baseline ---
    print("[*] Running Naive Baseline (scalar, single-threaded)...")
    t0 = time.perf_counter_ns()
    engine.upscale_catmullrom_naive(
        in_img, in_w, in_h, in_c, out_naive, out_w, out_h)
    t1 = time.perf_counter_ns()
    naive_ms = (t1 - t0) / 1e6
    naive_mpx = (total_out_pixels / 1e6) / (naive_ms / 1e3)
    print(f"    Time       : {naive_ms:.2f} ms")
    print(f"    Throughput : {naive_mpx:.2f} Megapixels/sec\n")

    # --- Optimized engine ---
    print("[*] Running Optimized Engine (AVX2 + FMA + Tiling + jthread)...")
    t0 = time.perf_counter_ns()
    engine.upscale_catmullrom_optimized(
        in_img, in_w, in_h, in_c, out_opt, out_w, out_h)
    t1 = time.perf_counter_ns()
    opt_ms  = (t1 - t0) / 1e6
    opt_mpx = (total_out_pixels / 1e6) / (opt_ms / 1e3)
    print(f"    Time       : {opt_ms:.2f} ms")
    print(f"    Throughput : {opt_mpx:.2f} Megapixels/sec\n")

    # --- Correctness check ---
    max_diff = np.max(np.abs(out_naive - out_opt))
    print(f"[*] Max absolute difference (naive vs opt): {max_diff:.6f}")
    if max_diff > 1.0:
        print("    WARNING: outputs diverge significantly!")

    # --- Speedup ---
    speedup = naive_ms / opt_ms
    print(f"\n{'=' * 60}")
    print(f"  Speedup Factor : {speedup:.2f}x")
    print(f"  Naive          : {naive_ms:>10.2f} ms  ({naive_mpx:>8.2f} MPx/s)")
    print(f"  Optimized      : {opt_ms:>10.2f} ms  ({opt_mpx:>8.2f} MPx/s)")
    print(f"{'=' * 60}")

    # --- Save output images ---
    cv2.imwrite("input_test.jpg", to_hwc(in_img))
    cv2.imwrite("output_naive.jpg", to_hwc(out_naive))
    cv2.imwrite("output_optimized.jpg", to_hwc(out_opt))
    print("\n[+] Saved: input_test.jpg, output_naive.jpg, output_optimized.jpg")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AuraScale Benchmark Suite")
    parser.add_argument("--input", "-i", type=str, default=None,
                        help="Path to input image (default: test_image.jpg)")
    parser.add_argument("--scale", type=float, default=30.0,
                        help="Upscale factor (default: 30)")
    args = parser.parse_args()

    benchmark(args.scale, args.input)
