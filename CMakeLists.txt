cmake_minimum_required(VERSION 3.22)
project(cuDeep LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ---- Options ----
option(CUDEEP_BUILD_TESTS "Build unit tests" ON)
option(CUDEEP_BUILD_BENCHMARKS "Build benchmarks" ON)
option(CUDEEP_BUILD_PYTHON "Build Python bindings" ON)

# ---- CUDA setup ----
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;87;89;90" CACHE STRING "CUDA architectures")

# Aggressive optimisation — --use_fast_math enables __expf, __logf, fused
# multiply-add, and denormal-flush-to-zero.  --extra-device-vectorization
# tells ptxas to try harder on register allocation / ILP.
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --extra-device-vectorization")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda --expt-relaxed-constexpr")

# ---- Core library sources ----
set(CUDEEP_SOURCES
    src/tensor.cu
    src/memory.cu
    src/stream.cu
    src/kernels/elementwise_kernel.cu
    src/kernels/matmul_kernel.cu
    src/kernels/activation_kernel.cu
    src/kernels/conv_kernel.cu
    src/kernels/loss_kernel.cu
    src/kernels/reduce_kernel.cu
    src/kernels/optim_kernel.cu
    src/kernels/pool_kernel.cu
    src/kernels/norm_kernel.cu
    src/kernels/unary_kernel.cu
)

add_library(cudeep SHARED ${CUDEEP_SOURCES})

target_include_directories(cudeep PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudeep PUBLIC
    CUDA::cudart
    CUDA::cublas
    CUDA::curand
)

set_target_properties(cudeep PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# ---- Python bindings (pybind11) ----
if(CUDEEP_BUILD_PYTHON)
    find_package(pybind11 CONFIG QUIET)
    if(NOT pybind11_FOUND)
        find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
        execute_process(
            COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
            OUTPUT_VARIABLE pybind11_DIR
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    pybind11_add_module(_cudeep_core src/bindings/module.cpp)
    target_link_libraries(_cudeep_core PRIVATE cudeep)
    target_include_directories(_cudeep_core PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    install(TARGETS _cudeep_core
        LIBRARY DESTINATION ${Python3_SITEARCH}/cuDeep
    )
endif()

# ---- Benchmarks ----
if(CUDEEP_BUILD_BENCHMARKS)
    find_library(CUDNN_LIB cudnn HINTS ${CUDAToolkit_LIBRARY_DIR} /usr/lib/aarch64-linux-gnu /usr/lib/x86_64-linux-gnu)
    find_path(CUDNN_INCLUDE cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS} /usr/include)

    if(CUDNN_LIB AND CUDNN_INCLUDE)
        add_executable(bench_vs_reference benchmarks/bench_vs_reference.cu)
        target_link_libraries(bench_vs_reference PRIVATE cudeep CUDA::cublas ${CUDNN_LIB})
        target_include_directories(bench_vs_reference PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include ${CUDNN_INCLUDE})
        set_target_properties(bench_vs_reference PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
        message(STATUS "cuDNN found: ${CUDNN_LIB} — bench_vs_reference enabled")
    else()
        message(WARNING "cuDNN not found — bench_vs_reference will NOT be built")
    endif()

    add_executable(bench_sgemm benchmarks/bench_sgemm.cu)
    target_link_libraries(bench_sgemm PRIVATE cudeep CUDA::cublas)
    target_include_directories(bench_sgemm PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    set_target_properties(bench_sgemm PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/benchmarks)
endif()

# ---- Tests ----
if(CUDEEP_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---- Install ----
install(TARGETS cudeep
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/cudeep DESTINATION include)
