cmake_minimum_required(VERSION 3.17)
project(final CUDA)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda")

include_directories(
        ndarray autodiff
)

set(NDARRAY_SOURCES
        ndarray/ndarray.cu
        ndarray/ndarray_kernel.cu
)

set(AUTODIFF_SOURCES
        autodiff/operators.cu
        autodiff/autodiff.cu
)

set(UTILS_SOURCES
        utils/global_cublas_handle.cu
)

add_executable(final
        main.cu
        global_curand_generator.cuh
        ${NDARRAY_SOURCES}
        ${AUTODIFF_SOURCES}
        ${UTILS_SOURCES}
        tensor.cu
)

set_target_properties(final PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(final PRIVATE cublas curand)

set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)
include_directories(${pybind11_INCLUDE_DIR})

#link_directories(/opt/miniconda3/envs/aiprog/lib) # Align with conda env lib, not system lib
#pybind11_add_module(Genshin tensor_module.cu ndarray/ndarray.cu)
#target_link_libraries(Genshin PRIVATE cublas curand)
#install(TARGETS Genshin DESTINATION .)