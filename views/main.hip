#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cmath>
#include <iostream>

#define HIP_CHECK(call) \
    do { \
        hipError_t err_ = (call); \
        if (err_ != hipSuccess) { \
            std::fprintf(stderr, "HIP Error: %s (%s:%d)\n", \
                hipGetErrorString(err_), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while(0) \

int main() {
    std::cout << "Everything is ok" << '\n';

    return 0;
}