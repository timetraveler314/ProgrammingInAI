project(hw2 CUDA)
cmake_minimum_required(VERSION 3.17)

set(CMAKE_CXX_STANDARD 20)

add_executable(hw2 main.cu tensor.cu
        tensornn.cuh)

set_target_properties(hw2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(hw2 PRIVATE cublas)