cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(cs521_mp1 LANGUAGES CXX)

find_package(OpenMP REQUIRED)

add_executable(mp1_cpu gemm_cpu.cpp)
target_link_libraries(mp1_cpu PRIVATE OpenMP::OpenMP_CXX)

target_compile_options(mp1_cpu PRIVATE
    -O3
    -march=native
    -ffast-math
    -fomit-frame-pointer
)
