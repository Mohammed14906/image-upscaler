cmake_minimum_required(VERSION 3.15)
project(ImageUpscaler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(upscale_engine SHARED src/upscale_engine.cpp)

target_include_directories(upscale_engine PRIVATE src)

# Enable AVX2, FMA, and aggressive optimizations
if(MSVC)
    target_compile_options(upscale_engine PRIVATE /arch:AVX2 /O2 /fp:fast)
else()
    target_compile_options(upscale_engine PRIVATE
        -O3 -march=native -mavx2 -mfma -ffast-math)
    if(WIN32)
        target_link_options(upscale_engine PRIVATE
            -static -static-libgcc -static-libstdc++)
    endif()
endif()
