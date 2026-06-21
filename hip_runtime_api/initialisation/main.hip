#include <hip/hip_runtime.h>

#include <iostream>

int main() {
    int deviceCount;
    if (hipGetDeviceCount(&deviceCount) == hipSuccess) {
        for (int i = 0; i < deviceCount; ++i) {
            hipDeviceProp_t prop;
            if (hipGetDeviceProperties(&prop, i) == hipSuccess) {
                std::cout << "Device" << i << prop.name << std::endl;
            }
        }
    }

    return 0;
}
