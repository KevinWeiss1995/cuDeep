#pragma once

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace cudeep {

#define CUDEEP_CHECK_CUDA(call)                                              \
    do {                                                                     \
        cudaError_t err = (call);                                            \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(                                        \
                std::string("CUDA error at ") + __FILE__ + ":" +             \
                std::to_string(__LINE__) + " — " +                           \
                cudaGetErrorString(err));                                     \
        }                                                                    \
    } while (0)

#define CUDEEP_CHECK_LAST_KERNEL()                                           \
    do {                                                                     \
        cudaError_t err = cudaGetLastError();                                \
        if (err != cudaSuccess) {                                            \
            throw std::runtime_error(                                        \
                std::string("Kernel launch error at ") + __FILE__ + ":" +    \
                std::to_string(__LINE__) + " — " +                           \
                cudaGetErrorString(err));                                     \
        }                                                                    \
    } while (0)

#define CUDEEP_ASSERT(cond, msg)                                             \
    do {                                                                     \
        if (!(cond)) {                                                       \
            throw std::runtime_error(                                        \
                std::string("cuDeep assertion failed: ") + (msg) +           \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));          \
        }                                                                    \
    } while (0)

}  // namespace cudeep
