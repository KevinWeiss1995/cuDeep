find_package(CUDAToolkit REQUIRED)

add_executable(test_tensor unit/test_tensor.cu)
target_link_libraries(test_tensor PRIVATE cudeep CUDA::cudart CUDA::curand)
target_include_directories(test_tensor PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_tensor COMMAND test_tensor)

add_executable(test_matmul unit/test_matmul.cu)
target_link_libraries(test_matmul PRIVATE cudeep CUDA::cudart)
target_include_directories(test_matmul PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_matmul COMMAND test_matmul)

add_executable(test_kernels unit/test_kernels.cu)
target_link_libraries(test_kernels PRIVATE cudeep CUDA::cudart)
target_include_directories(test_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_kernels COMMAND test_kernels)

add_executable(test_advanced_kernels unit/test_advanced_kernels.cu)
target_link_libraries(test_advanced_kernels PRIVATE cudeep CUDA::cudart)
target_include_directories(test_advanced_kernels PRIVATE ${CMAKE_SOURCE_DIR}/include)
add_test(NAME test_advanced_kernels COMMAND test_advanced_kernels)
