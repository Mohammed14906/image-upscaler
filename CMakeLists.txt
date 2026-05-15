cmake_minimum_required(VERSION 3.15)
project(AuraScale LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(aurascale_kernel SHARED src/aurascale_kernel.cpp)

target_include_directories(aurascale_kernel PRIVATE src)

# Enable AVX2, FMA, and aggressive optimizations
if(MSVC)
    target_compile_options(aurascale_kernel PRIVATE /arch:AVX2 /O2 /fp:fast)
else()
    target_compile_options(aurascale_kernel PRIVATE
        -O3 -march=native -mavx2 -mfma -ffast-math)
    if(WIN32)
        target_link_options(aurascale_kernel PRIVATE
            -static -static-libgcc -static-libstdc++)
    endif()
endif()
