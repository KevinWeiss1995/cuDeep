cmake_minimum_required(VERSION 3.24)
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

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86;89;90" CACHE STRING "CUDA architectures")

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
)

add_library(cudeep SHARED ${CUDEEP_SOURCES})

target_include_directories(cudeep PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(cudeep PUBLIC
    CUDA::cudart
    CUDA::cublas
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
