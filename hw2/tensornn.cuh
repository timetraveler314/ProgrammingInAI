//
// Created by timetraveler314 on 10/28/24.
//

#ifndef TENSORNN_CUH
#define TENSORNN_CUH

#include "tensor.h"
#include "tensor_kernel.h"

namespace TensorNN {
    inline Tensor forward_fc(const Tensor &input, const Tensor &weight, const Tensor &bias) {
        // Shape check
        if (input.getShape().size() != 2 || weight.getShape().size() != 2 || bias.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for forward_fc");
        }

        const int N = input.getShape()[0];
        const int Cin = input.getShape()[1];
        const int Cout = weight.getShape()[0];

        if (Cin != weight.getShape()[1] || Cout != bias.getShape()[0]) {
            throw std::runtime_error("Invalid shape for forward_fc");
        }

        auto [device, x, w, b] = Tensor::unifyDevice(input, weight, bias);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for forward_fc");
        }

        Tensor y({N, Cout}, TensorDevice::GPU);
        tensor_kernel::forward_fc_kernel_gpu(x.getRawData(), w.getRawData(), b.getRawData(), y.getRawData(), N, Cin, Cout);
        return y;
    }

    inline std::tuple<Tensor, Tensor, Tensor> backward_fc(const Tensor &dy, const Tensor &input, const Tensor &weight) {
        // Shape check.
        if (dy.getShape().size() != 2 || input.getShape().size() != 2 || weight.getShape().size() != 2) {
            throw std::runtime_error("Invalid shape for backward_fc");
        }

        const int N = input.getShape()[0];
        const int Cin = input.getShape()[1];
        const int Cout = weight.getShape()[0];

        if (dy.getShape()[0] != N || dy.getShape()[1] != Cout || Cin != weight.getShape()[1]) {
            throw std::runtime_error("Invalid shape for backward_fc");
        }

        auto [device, x, w, b] = Tensor::unifyDevice(input, weight, Tensor({1}, TensorDevice::CPU));

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for backward_fc");
        }

        Tensor dx({N, Cin}, TensorDevice::GPU), dw({Cout, Cin}, TensorDevice::GPU), db({Cout}, TensorDevice::GPU);
        tensor_kernel::backward_fc_kernel_gpu(x.getRawData(), w.getRawData(), dy.getRawData(), dx.getRawData(), dw.getRawData(), db.getRawData(), N, Cin, Cout);
        return {dx, dw, db};
    }

    inline Tensor forward_max_pooling_2x2(const Tensor &input) {
        if (input.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for forward_max_pooling_2x2");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];
        const int H = input.getShape()[2];
        const int W = input.getShape()[3];

        auto [device, x] = Tensor::unifyDevice(input);

        Tensor y({N, C, H / 2, W / 2}, TensorDevice::GPU);

        cudaMemset(y.getRawData(), 0, y.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            tensor_kernel::forward_max_pooling_2x2_kernel_gpu<<<CudaGetBlocks(C * (H / 2) * (W / 2)), kCudaThreadsNum>>>(x.getRawData() + i * C * H * W, y.getRawData() + i * C * H / 2 * W / 2, C, H, W);
        }

        return y;
    }

    inline Tensor backward_max_pooling_2x2(const Tensor & upstream_grad, const Tensor &input) {
        if (upstream_grad.getShape().size() != 4 || input.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for backward_max_pooling_2x2");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];
        const int H = input.getShape()[2];
        const int W = input.getShape()[3];

        if (upstream_grad.getShape()[0] != N || upstream_grad.getShape()[1] != C || upstream_grad.getShape()[2] != H / 2 || upstream_grad.getShape()[3] != W / 2) {
            throw std::runtime_error("Invalid shape for backward_max_pooling_2x2");
        }

        auto [device, dy, x] = Tensor::unifyDevice(upstream_grad, input);
        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for backward_max_pooling_2x2");
        }

        Tensor dx({N, C, H, W}, TensorDevice::GPU);

        cudaMemset(dx.getRawData(), 0, dx.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            tensor_kernel::backward_max_pooling_2x2_kernel_gpu<<<CudaGetBlocks(C * (H / 2) * (W / 2)), kCudaThreadsNum>>>(
                dy.getRawData() + i * C * H / 2 * W / 2,
                x.getRawData() + i * C * H * W,
                dx.getRawData() + i * C * H * W,
                C, H, W
            );
        }

        return dx;
    }

    inline Tensor forward_softmax(const Tensor &input) {
        if (input.getShape().size() != 2) {
            throw std::runtime_error("Invalid shape for forward_softmax");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];

        auto [device, x] = Tensor::unifyDevice(input);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for forward_softmax");
        }

        Tensor y({N, C}, TensorDevice::GPU);
        tensor_kernel::forward_softmax_kernel_gpu(x.getRawData(), y.getRawData(), N, C);

        return y;
    }

    inline TensorDataType cross_entropy(const Tensor &input, const Tensor &ground_truth) {
        if (input.getShape().size() != 2 || ground_truth.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for cross_entropy");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];

        auto [device, x, gt] = Tensor::unifyDevice(input, ground_truth);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for cross_entropy");
        }

        TensorDataType *d_output_loss;
        cudaMalloc(&d_output_loss, sizeof(TensorDataType));
        cudaMemset(d_output_loss, 0, sizeof(TensorDataType));

        tensor_kernel::cross_entropy_kernel_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(x.getRawData(), gt.getRawData(), d_output_loss, N, C);

        TensorDataType output_loss = 0.0f;
        cudaMemcpy(&output_loss, d_output_loss, sizeof(TensorDataType), cudaMemcpyDeviceToHost);
        cudaFree(d_output_loss);

        return output_loss;
    }

    // The `input` argument here is the output of the softmax layer.
    inline Tensor backward_softmax_cross_entropy(const Tensor &softmax_output, const Tensor &ground_truth) {
        if (softmax_output.getShape().size() != 2 || ground_truth.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for backward_softmax_cross_entropy");
        }

        const int N = softmax_output.getShape()[0];
        const int C = softmax_output.getShape()[1];

        auto [device, x, gt] = Tensor::unifyDevice(softmax_output, ground_truth);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for backward_softmax_cross_entropy");
        }

        Tensor dx({N, C}, TensorDevice::GPU);
        tensor_kernel::backward_softmax_cross_entropy_kernel_gpu<<<CudaGetBlocks(N * C), kCudaThreadsNum>>>(x.getRawData(), gt.getRawData(), dx.getRawData(), N, C);

        return dx;
    }

    inline Tensor conv2d_3x3(const Tensor &images, const Tensor &kernels) {
        if (images.getShape().size() != 4 || kernels.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for conv2d_3x3");
        }

        const int N = images.getShape()[0];
        const int C = images.getShape()[1];
        const int H = images.getShape()[2];
        const int W = images.getShape()[3];

        const int K = kernels.getShape()[0];
        const int K_C = kernels.getShape()[1];
        const int K_H = kernels.getShape()[2];
        const int K_W = kernels.getShape()[3];

        if (K_C != C || K_H != 3 || K_W != 3) {
            throw std::runtime_error("Invalid shape for conv2d_3x3");
        }

        auto [device, im, k] = Tensor::unifyDevice(images, kernels);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for conv2d_3x3");
        }

        cublasHandle_t handle;
        cublasCreate(&handle);

        Tensor y({N, K, H, W}, TensorDevice::GPU);

        TensorDataType *col;
        cudaMalloc(&col, C * 9 * H * W * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            tensor_kernel::im2col_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                im.getRawData() + i * C * H * W, // Current image
                col,
                C, H, W, // image C, H, W
                3, 3, // kernel size
                1, 1, // padding
                1, 1, // stride
                H, W // col H, W
            );

            cudaDeviceSynchronize();

            tensor_kernel::gemm_row_major_gpu(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                K, H * W, C * 9,
                1.0f, 0.0f,
                k.getRawData(), col, y.getRawData() + i * K * H * W
            );
        }

        cudaFree(col);
        cublasDestroy(handle);

        return y;
    }

    inline std::tuple<Tensor, Tensor> conv2d_3x3_backward(const Tensor& images, const Tensor& kernels, const Tensor& output_grad) {
        if (images.getShape().size() != 4 || kernels.getShape().size() != 4 || output_grad.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for conv2d_3x3 backward");
        }

        const int N = images.getShape()[0];
        const int C = images.getShape()[1];
        const int H = images.getShape()[2];
        const int W = images.getShape()[3];

        const int K = kernels.getShape()[0];
        const int K_C = kernels.getShape()[1];
        const int K_H = kernels.getShape()[2];
        const int K_W = kernels.getShape()[3];

        if (K_C != C || K_H != 3 || K_W != 3) {
            throw std::runtime_error("Invalid shape for conv2d_3x3 backward");
        }

        if (output_grad.getShape()[0] != N || output_grad.getShape()[1] != K || output_grad.getShape()[2] != H || output_grad.getShape()[3] != W) {
            throw std::runtime_error("Output gradient shape does not match expected dimensions");
        }

        auto [device, im, k] = Tensor::unifyDevice(images, kernels);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for conv2d_3x3 backward");
        }

        cublasHandle_t handle;
        cublasCreate(&handle);

        Tensor kernel_grad({K, C, 3, 3}, TensorDevice::GPU);
        TensorDataType *col;
        cudaMalloc(&col, C * 9 * H * W * sizeof(TensorDataType));

        cudaMemset(kernel_grad.getRawData(), 0, kernel_grad.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            tensor_kernel::im2col_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                im.getRawData() + i * C * H * W, // Current image
                col,
                C, H, W, // image C, H, W
                3, 3, // kernel size
                1, 1, // padding
                1, 1, // stride
                H, W // col H, W
            );

            cudaDeviceSynchronize();

            tensor_kernel::gemm_row_major_gpu(
                handle, CUBLAS_OP_N, CUBLAS_OP_T,
                K, C * 9, H * W,
                1.0f, 1.0f,
                output_grad.getRawData() + i * K * H * W, col, kernel_grad.getRawData()
            );
        }

        // No need to average the kernel gradients

        // Compute the input gradient

        Tensor input_grad({N, C, H, W}, TensorDevice::GPU);

        TensorDataType *grad_col;
        cudaMalloc(&grad_col, C * 9 * H * W * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            tensor_kernel::gemm_row_major_gpu(
                handle, CUBLAS_OP_T, CUBLAS_OP_N,
                C * 9, H * W, K,
                1.0f, 0.0f,
                k.getRawData(), output_grad.getRawData() + i * K * H * W, grad_col
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(("gemm_row_major_gpu" + std::to_string(i)).c_str());

            tensor_kernel::col2im_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                grad_col,
                input_grad.getRawData() + i * C * H * W,
                C, H, W,
                3, 3,
                1, 1,
                1, 1,
                H, W
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("col2im_kernel");
        }

        cudaFree(col);
        cublasDestroy(handle);

        return {input_grad, kernel_grad};
}

}

#endif //TENSORNN_CUH
