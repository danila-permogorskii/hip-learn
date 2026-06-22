#include <cstdio>
#include <hip/hip_runtime.h>

#define HIP_CHECK(call) \
    do { \
        hipError_t _err = (call); \
        if (_err != hipSuccess) { \
            std::fprintf(stderr, "HIP Error: %s (%s:%d)\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            std::exit(EXIT_FAILURE); \
        } \
    } while (0)


int main() {
    std::printf("Everything is ok\n");
}
