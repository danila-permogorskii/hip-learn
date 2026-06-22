cmake_minimum_required(VERSION 3.21)
project(ck_tile_course LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_HIP_STANDARD 20)
set(CMAKE_HIP_ARCHITECTURES gfx942 gfx950)        # MI300X / CDNA3 and MI350x / CDNA4
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)      # feeds clangd

# One target per experiment — lägg till en per övning
add_executable(toolchain_smoke 00-toolchain-smoke/toolchain_smoke.hip)
set_source_files_properties(
        00-toolchain-smoke/toolchain_smoke.hip PROPERTIES LANGUAGE HIP)

add_executable(views views/main.hip)
set_source_files_properties(
        views/main.hip PROPERTIES LANGUAGE HIP
)

add_executable(descriptors_and_transforms descriptors_and_transforms/main.hip)
set_source_files_properties(
        descriptors_and_transforms/main.hip PROPERTIES LANGUAGE HIP
)
