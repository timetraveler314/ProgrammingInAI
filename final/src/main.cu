#include <iostream>

#include "ndarray/ndarray.h"

#include "tensornn.h"

#include "tensor.h"

int main() {
    auto rand_ndarr_cpu = NdArray::uniform({3, 3}, Device::GPU);
    auto rand_ndarr = NdArray::uniform({3, 3}, Device::GPU);

    std::cout << rand_ndarr_cpu << std::endl << rand_ndarr << std::endl;

    auto tensor = Tensor(rand_ndarr_cpu - rand_ndarr, true);
    auto relu_out = TensorNN::ReLU(tensor);

    std::cout << relu_out << std::endl;

    std::cout << "=== Autodiff ===" << std::endl;

    relu_out.backward(Tensor(NdArray::ones({3, 3}, Device::GPU)));

    std::cout << tensor.grad();
}

