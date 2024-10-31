cmake_minimum_required(VERSION 3.17)
project(hw2 CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

add_executable(hw2 main.cu tensor.cu
        tensornn.cuh
        global_curand_generator.cuh)

set_target_properties(hw2 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(hw2 PRIVATE cublas curand)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

link_directories(/opt/miniconda3/envs/aiprog/lib) # Align with conda env lib, not system lib
pybind11_add_module(Genshin tensor_module.cu tensor.cu)
target_link_libraries(Genshin PRIVATE cublas curand)
install(TARGETS Genshin DESTINATION .)