cmake_minimum_required(VERSION 3.24)
project(mandelbrot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

# FMA contraction makes GPU results differ slightly from the (non-FMA) CPU
# builds; configure with -DCUDA_FMAD=OFF for bit-identical CPU/GPU output.
option(CUDA_FMAD "Allow fused multiply-add in CUDA kernels" ON)

find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)
find_library(GMP_LIB gmp REQUIRED)
find_library(GMPXX_LIB gmpxx REQUIRED)

add_executable(mandelbrot
  src/main.cpp
  src/options.cpp
  src/reference.cpp
  src/cpu.cpp
  src/video.cpp
  src/gpu.cu
  ${CMAKE_CURRENT_SOURCE_DIR}/../cuda/mandelbrot.cu)
target_include_directories(mandelbrot PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../cuda)
target_compile_options(mandelbrot PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-O3 -Wall -Wextra>)
if(NOT CUDA_FMAD)
  target_compile_options(mandelbrot PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
endif()
target_link_libraries(mandelbrot PRIVATE Threads::Threads CUDA::cudart ${GMPXX_LIB} ${GMP_LIB})
