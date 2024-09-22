cmake_minimum_required(VERSION 3.17)
project(HW1 CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(hw1 main.cu tensor.cu)

set_target_properties(hw1 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)