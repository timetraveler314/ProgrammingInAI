#include <iostream>

#include "ndarray/ndarray.h"
#include "ndarray/ndarray_kernel.cuh"
#include <thrust/functional.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "global_curand_generator.cuh"
#include "ndarray/nn.cuh"

int main() {
    std::cout << "=== Test 1 : Fully connected layer ===" << std::endl;

    auto x = NdArray::iota({2, 3}, Device::GPU);
    auto w = NdArray::iota({4, 3}, Device::GPU);
    auto b = NdArray::iota({4}, Device::GPU);

    std::cout << "X: " << x << std::endl;
    std::cout << "W: " << w << std::endl;
    std::cout << "B: " << b << std::endl;

    auto result = NdArrayNN::forward_fc(x, w, b);

    std::cout << "Result: " << result << std::endl;

    auto [dx, dw, db] = NdArrayNN::backward_fc(NdArray::uniform({2,4}, Device::GPU), x, w);

    std::cout << "Backward gradients:" << std::endl;

    std::cout << "dx: " << dx << std::endl;
    std::cout << "dw: " << dw << std::endl;
    std::cout << "db: " << db << std::endl;

    std::cout << "=== Test 2 : Convolutional layer ===" << std::endl;

    std::cout << "== im2col test ==" << std::endl;

    int C = 1;
    auto im = NdArray::iota({C, 3, 4}, Device::GPU);
    auto col = NdArray({C, 9, 12}, Device::GPU);

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

    auto im2 = NdArray({C, 3, 4}, Device::GPU);

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

    auto ims = NdArray::iota({N,C,H,W}, Device::GPU);
    auto upstream_grad = NdArray::iota({N,K,H,W}, Device::GPU);
    // Random network parameters using cuRAND
    auto kernel = NdArray::uniform({K,C,3,3}, Device::GPU);

    auto [dims, dkernel] = NdArrayNN::conv2d_3x3_backward(ims, kernel, upstream_grad);

    std::cout << "dims: " << dims << std::endl;
    std::cout << "dkernel: " << dkernel << std::endl;

    std::cout << "=== Test 3 : Max Pooling 2x2 ===" << std::endl;

    auto im_pool = NdArray::iota({1, 1, 4, 4}, Device::GPU);
    auto pool_result = NdArrayNN::forward_max_pooling_2x2(im_pool);

    std::cout << "im_pool: " << im_pool << std::endl;
    std::cout << "pool_result: " << pool_result << std::endl;

    auto upstream_grad_pool = NdArray::uniform({1, 1, 2, 2}, Device::GPU);
    auto im_pool_grad = NdArrayNN::backward_max_pooling_2x2(upstream_grad_pool, im_pool);

    std::cout << "== max pooling backward test ==" << std::endl;
    std::cout << "upstream_grad_pool: " << upstream_grad_pool << std::endl;
    std::cout << "im_pool_grad: " << im_pool_grad << std::endl;

    std::cout << "=== Test 4 : Softmax (Forward) ===" << std::endl;

    auto x_softmax = NdArray::iota({2, 3}, Device::GPU);
    auto softmax_result = NdArrayNN::forward_softmax(x_softmax);

    std::cout << "x_softmax: " << x_softmax << std::endl;
    std::cout << "softmax_result: " << softmax_result << std::endl;

    std::cout << "=== Test 5 : Softmax (Backward) & Cross Entropy Loss ===" << std::endl;

    auto ground_truth = NdArray::uniform({2}, Device::GPU);
    auto ground_truth1 = NdArray::uniform({200}, Device::GPU);
    auto ground_truth2 = NdArray::uniform({211}, Device::GPU);
    auto ground_truth3 = NdArray::uniform({2213}, Device::GPU);
    auto ground_truth4 = NdArray::uniform({21}, Device::GPU);
    auto ground_truth5 = NdArray::uniform({2}, Device::GPU);
    auto ground_truth6 = NdArray::uniform({2}, Device::GPU);
    auto ground_truth7 = NdArray::uniform({2}, Device::GPU);
    std::cout << "ground_truth7: " << ground_truth1 << std::endl;
    auto loss = NdArrayNN::cross_entropy(softmax_result, ground_truth);

    std::cout << "ground_truth: " << ground_truth << std::endl;
    std::cout << "loss: " << loss << std::endl;

    auto dx_softmax = NdArrayNN::backward_softmax_cross_entropy(softmax_result, ground_truth);

    std::cout << "== Test Combined Softmax Backward and Cross Entropy Loss ==" << std::endl;
    std::cout << "dx_softmax: " << dx_softmax << std::endl;

    return 0;
}

