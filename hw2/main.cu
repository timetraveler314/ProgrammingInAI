#include <iostream>

#include "tensor.h"
#include "tensor_kernel.h"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "global_curand_generator.cuh"
#include "tensornn.cuh"

Tensor get_test_neg1_1_tensor();

int main() {
    auto x = Tensor::iota({2, 3}, TensorDevice::GPU);
    auto w = Tensor::iota({4, 3}, TensorDevice::GPU);
    auto b = Tensor::iota({4}, TensorDevice::GPU);

    std::cout << "X: " << x << std::endl;
    std::cout << "W: " << w << std::endl;
    std::cout << "B: " << b << std::endl;

    auto result = TensorNN::forward_fc(x, w, b);

    std::cout << "Result: " << result << std::endl;

    auto [dx, dw, db] = TensorNN::backward_fc(get_test_neg1_1_tensor(), x, w);

    std::cout << "dx: " << dx << std::endl;
    std::cout << "dw: " << dw << std::endl;
    std::cout << "db: " << db << std::endl;

    auto rand_tensor = Tensor::uniform({2,3,4}, TensorDevice::GPU);

    std::cout << rand_tensor << std::endl;

    std::cout << "im2col test" << std::endl;

    // 3 times 3 conv kernel

    const int C = 1;
    auto im = Tensor::iota({C, 3, 4}, TensorDevice::GPU);
    auto col = Tensor({C, 9, 12}, TensorDevice::GPU);

    tensor_kernel::im2col_kernel<<<CudaGetBlocks(C * 3 * 4), kCudaThreadsNum>>>(
        im.getRawData(),
        col.getRawData(),
        C, 3, 4, // image C, H, W
        3, 3, // kernel size
        1, 1, // padding
        1, 1, // stride
        3, 4 // col H, W
    );

    std::cout << im << std::endl;

    std::cout << col << std::endl;

    auto im2 = Tensor({C, 3, 4}, TensorDevice::GPU);

    tensor_kernel::col2im_kernel<<<CudaGetBlocks(C * 3 * 4), kCudaThreadsNum>>>(
        col.getRawData(),
        im2.getRawData(),
        C, 3, 4, // image C, H, W
        3, 3, // kernel size
        1, 1, // padding
        1, 1, // stride
        3, 4 // col H, W
    );

    std::cout << im2 << std::endl;

    return 0;
}

Tensor random_gpu_tensor(const std::vector<int>& shape) {
    Tensor t(shape, TensorDevice::CPU);


    return t.gpu();
}

Tensor get_test_neg1_1_tensor() {
    Tensor t1({2,4}, TensorDevice::CPU);

    for (int i = 0; i < t1.size(); i++) {
        t1.getRawData()[i] = i % 2 == 0 ? -1 : 1;
    }

    return t1.gpu();
}
