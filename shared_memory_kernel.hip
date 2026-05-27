#include <hip/hip_runtime.h>

#include <cstdlib>
#include <iostream>

extern __shared__ int shared_array[];

__global__ void kernel() {
    // initialise shared memory
    shared_array[threadIdx.x] = threadIdx.x;
    // use shared memory - synchronise to make sure, that all threads of the
    // block see all changes to shared memory
    __syncthreads();
}

int main() {

    // shared memory inthis case depends on the configurable block size
    constexpr int blockSize = 256;
    constexpr int sharedMemSize = blockSize * sizeof(int);
    constexpr int gridSize = 2;

    kernel<<<dim3(gridSize), dim3(blockSize), sharedMemSize, 0>>>();
    if (auto err = hipDeviceSynchronize(); err != hipSuccess)
        std::cerr << "HIP error" << err << ": " << hipGetErrorString(err) << std::endl;

    return EXIT_SUCCESS;
}