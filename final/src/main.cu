#include <iostream>

#include "ndarray/ndarray.h"
#include "ndarray/ndarray_kernel.cuh"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "global_curand_generator.cuh"
#include "ndarray/nn.cuh"
#include "tensornn.h"

#include "tensor.h"

int main() {
    auto rand_ndarr_cpu = NdArray::uniform({3, 3}, Device::GPU);
    auto rand_ndarr = NdArray::uniform({3, 3}, Device::GPU);

    std::cout << rand_ndarr_cpu << std::endl << rand_ndarr << std::endl;

    auto tensor = Tensor(rand_ndarr_cpu - rand_ndarr);
    auto relu_out = TensorNN::ReLU(tensor);

    std::cout << relu_out << std::endl;
}

