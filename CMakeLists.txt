cmake_minimum_required(VERSION 3.21)
project(ck_tile_course LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx942)        # MI300X / CDNA3
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)      # feeds clangd

# One target per experiment — lägg till en per övning
add_executable(toolchain_smoke 00-toolchain-smoke/toolchain_smoke.hip)
add_executable(hip_runtime_api_init hip_runtime_api/initialisation/main.hip)
add_executable(hip_runtime_api_pageable hip_runtime_api/memory_management/pageable.hip)
add_executable(hip_runtime_api_pinned hip_runtime_api/memory_management/pinned.hip)
add_executable(matmul matmul/main.hip)
add_executable(sample sample/sample.hip)
add_executable(tile_distribution TileDistribution/main.hip)
set_source_files_properties(
  00-toolchain-smoke/toolchain_smoke.hip PROPERTIES LANGUAGE HIP)