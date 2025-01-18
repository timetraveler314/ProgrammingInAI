#include <iostream>

#include "tensor.h"
#include "tensor_kernel.h"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "global_curand_generator.cuh"
#include "tensornn.cuh"

int main() {
    std::cout << "=== Test 1 : Fully connected layer ===" << std::endl;

    auto x = Tensor::iota({2, 3}, TensorDevice::GPU);
    auto w = Tensor::iota({4, 3}, TensorDevice::GPU);
    auto b = Tensor::iota({4}, TensorDevice::GPU);

    std::cout << "X: " << x << std::endl;
    std::cout << "W: " << w << std::endl;
    std::cout << "B: " << b << std::endl;

    auto result = TensorNN::forward_fc(x, w, b);

    std::cout << "Result: " << result << std::endl;

    auto [dx, dw, db] = TensorNN::backward_fc(Tensor::uniform({2,4}, TensorDevice::GPU), x, w);

    std::cout << "Backward gradients:" << std::endl;

    std::cout << "dx: " << dx << std::endl;
    std::cout << "dw: " << dw << std::endl;
    std::cout << "db: " << db << std::endl;

    std::cout << "=== Test 2 : Convolutional layer ===" << std::endl;

    std::cout << "== im2col test ==" << std::endl;

    int C = 1;
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

    std::cout << "image: " << im << std::endl;

    std::cout << "col:" << col << std::endl;

    auto im2 = Tensor({C, 3, 4}, TensorDevice::GPU);

    std::cout << "== col2im test, using col from im2col test ==" << std::endl;

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

    std::cout << "== conv backward test ==" << std::endl;

    C = 2;
    int N = 1, H = 4, W = 3, K = 4;

    auto ims = Tensor::iota({N,C,H,W}, TensorDevice::GPU);
    auto upstream_grad = Tensor::iota({N,K,H,W}, TensorDevice::GPU);
    // Random network parameters using cuRAND
    auto kernel = Tensor::uniform({K,C,3,3}, TensorDevice::GPU);

    auto [dims, dkernel] = TensorNN::conv2d_3x3_backward(ims, kernel, upstream_grad);

    std::cout << "dims: " << dims << std::endl;
    std::cout << "dkernel: " << dkernel << std::endl;

    std::cout << "=== Test 3 : Max Pooling 2x2 ===" << std::endl;

    auto im_pool = Tensor::iota({1, 1, 4, 4}, TensorDevice::GPU);
    auto pool_result = TensorNN::forward_max_pooling_2x2(im_pool);

    std::cout << "im_pool: " << im_pool << std::endl;
    std::cout << "pool_result: " << pool_result << std::endl;

    auto upstream_grad_pool = Tensor::uniform({1, 1, 2, 2}, TensorDevice::GPU);
    auto im_pool_grad = TensorNN::backward_max_pooling_2x2(upstream_grad_pool, im_pool);

    std::cout << "== max pooling backward test ==" << std::endl;
    std::cout << "upstream_grad_pool: " << upstream_grad_pool << std::endl;
    std::cout << "im_pool_grad: " << im_pool_grad << std::endl;

    std::cout << "=== Test 4 : Softmax (Forward) ===" << std::endl;

    auto x_softmax = Tensor::iota({2, 3}, TensorDevice::GPU);
    auto softmax_result = TensorNN::forward_softmax(x_softmax);

    std::cout << "x_softmax: " << x_softmax << std::endl;
    std::cout << "softmax_result: " << softmax_result << std::endl;

    std::cout << "=== Test 5 : Softmax (Backward) & Cross Entropy Loss ===" << std::endl;

    auto ground_truth = Tensor::uniform({2}, TensorDevice::GPU);
    auto ground_truth1 = Tensor::uniform({200}, TensorDevice::GPU);
    auto ground_truth2 = Tensor::uniform({211}, TensorDevice::GPU);
    auto ground_truth3 = Tensor::uniform({2213}, TensorDevice::GPU);
    auto ground_truth4 = Tensor::uniform({21}, TensorDevice::GPU);
    auto ground_truth5 = Tensor::uniform({2}, TensorDevice::GPU);
    auto ground_truth6 = Tensor::uniform({2}, TensorDevice::GPU);
    auto ground_truth7 = Tensor::uniform({2}, TensorDevice::GPU);
    std::cout << "ground_truth7: " << ground_truth1 << std::endl;
    auto loss = TensorNN::cross_entropy(softmax_result, ground_truth);

    std::cout << "ground_truth: " << ground_truth << std::endl;
    std::cout << "loss: " << loss << std::endl;

    auto dx_softmax = TensorNN::backward_softmax_cross_entropy(softmax_result, ground_truth);

    std::cout << "== Test Combined Softmax Backward and Cross Entropy Loss ==" << std::endl;
    std::cout << "dx_softmax: " << dx_softmax << std::endl;

    return 0;
}

