#include <iostream>

#include "tensor.h"
#include "tensor_kernel.h"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "tensornn.cuh"

Tensor random_gpu_tensor(const std::vector<int>& shape);
Tensor get_test_neg1_1_tensor();

int main() {
    srand(42); // seed random number generator to allow for reproducing the results.

    auto x = Tensor({2, 3}, TensorDevice::CPU);
    auto w = Tensor({4, 3}, TensorDevice::CPU);
    auto b = Tensor({4}, TensorDevice::CPU);

    for (int i = 0; i < x.size(); i++) {
        x.data->space[i] = i;
    }
    for (int i = 0; i < w.size(); i++) {
        w.data->space[i] = i;
    }
    for (int i = 0; i < b.size(); i++) {
       b.data->space[i] = i;
    }

    auto xg = x.gpu();
    auto wg = w.gpu();
    auto bg = b.gpu();

    std::cout << "X: " << x << std::endl;
    std::cout << "W: " << w << std::endl;
    std::cout << "B: " << b << std::endl;

    auto result = TensorNN::forward_fc(xg, wg, bg);

    std::cout << "Result: " << result << std::endl;

    auto [dx, dw, db] = TensorNN::backward_fc(get_test_neg1_1_tensor(), xg, wg);

    std::cout << "dx: " << dx << std::endl;
    std::cout << "dw: " << dw << std::endl;
    std::cout << "db: " << db << std::endl;

    return 0;
}

Tensor random_gpu_tensor(const std::vector<int>& shape) {
    Tensor t(shape, TensorDevice::CPU);


    return t.gpu();
}

Tensor get_test_neg1_1_tensor() {
    Tensor t1({2,4}, TensorDevice::CPU);

    for (int i = 0; i < t1.size(); i++) {
        t1.data->space[i] = i % 2 == 0 ? -1 : 1;
    }

    return t1.gpu();
}
