# AuraScale 🚀

**AuraScale** is a high-throughput, modular image upscaling library utilizing a C++/Python hybrid architecture. It is designed to act as a high-performance preprocessing node for real-time AI vision pipelines (e.g., Lipsync, Inference) where native Python resizing is too slow for 4K/60FPS or extreme 8K scaling.

AuraScale uses a **Catmull-Rom (Mitchell-Netravali B=0, C=0.5)** spline interpolation kernel. The intrinsic negative lobes of this BC-Spline mathematically sharpen edges during the interpolation process itself, providing high-pass enhancement without the computational overhead of a separate sharpening pass.

## ⚡ Performance & Optimization Stack

AuraScale is aggressively optimized to run on modern CPUs, bypassing typical bottlenecks in high-resolution image processing:

- **AVX2 & FMA Vectorization**: The engine calculates 8 neighboring pixel weights and performs Fused Multiply-Add accumulations simultaneously using 256-bit SIMD registers.
- **L1 Cache Residency (Tiling)**: The output is computed in $256 \times 256$ pixel blocks to ensure the required input working set remains "hot" in the L1 Data Cache, practically eliminating memory starvation.
- **Planar RGB Processing**: Image data is converted from standard Interleaved (HWC) to Planar (CHW) format. This allows pure Stride-1 sequential memory loads.
- **Manual Loop Unrolling**: The inner $4 \times 4$ kernel loops are unrolled by a factor of 4, keeping the CPU instruction pipeline saturated and eliminating branch prediction failures.
- **Horizontal Stripping**: C++20 `std::jthread` maps horizontal strips of the image across all available hardware threads, ensuring 100% CPU utilization without thread locking overhead.

### Benchmarks (30x Scale, 256x256 -> 7680x7680, RGB)

| Implementation | Time | Throughput |
| --- | --- | --- |
| Naive (Scalar, Single-thread) | ~1744 ms | 101.45 MPx/s |
| **AuraScale Engine** | **~79 ms** | **2229.75 MPx/s** |
| **Speedup** | **~22x** | - |

## 🛠️ Build Instructions

### Prerequisites
- Python 3.8+
- CMake 3.15+
- A C++20 compatible compiler with AVX2 support (e.g., Clang, GCC, MSVC)

### Compiling the Engine
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release
```

## 🚀 Usage

The project provides a lightweight, dependency-free wrapper (`ctypes`) allowing seamless integration with NumPy and OpenCV.

Run the provided Python benchmarking suite:

```bash
# Run with the default test image at a massive 30x scale
python main.py --scale 30.0

# Process a specific image
python main.py --input /path/to/your/image.jpg --scale 4.0
```

## 📂 Project Structure

```text
├── CMakeLists.txt        # Build system configuration (-mavx2, -O3, etc.)
├── main.py               # Python hybrid interface & benchmarking suite
├── src/
│   ├── aurascale_kernel.hpp  # C ABI headers for ctypes integration
│   └── aurascale_kernel.cpp  # The heavily optimized AVX2/threaded engine
└── test_image.jpg        # Default input image for testing
```
