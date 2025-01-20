//
// Created by timetraveler314 on 10/28/24.
//

#ifndef NDARRAYNN_CUH
#define NDARRAYNN_CUH

#include <nn.cuh>

#include "ndarray.h"
#include "ndarray_kernel.cuh"
#include "../utils/global_cublas_handle.cuh"

/* NdArrayNN
 * This namespace contains the forward and backward functions for the neural network layers.
 * The forward functions take the input tensor and return the output tensor.
 * The backward functions take the input tensor and the gradient tensor and return a tuple of gradient tensors.
 */
namespace NdArrayNN {
    inline NdArray forward_relu(const NdArray &input) {
        // No shape check needed

        const int N = input.size();
        NdArray result(input.getShape(), input.getDevice());

        switch (input.getDevice()) {
            case Device::CPU:
                thrust::transform(thrust::host, input.getRawData(), input.getRawData() + N,
                    result.getRawData(),
                    [] (const TensorDataType &x) { return (x > 0) ? x : 0; });
                break;
            case Device::GPU:
                thrust::transform(thrust::device, input.getRawData(), input.getRawData() + N,
                    result.getRawData(),
                    [] __device__ (const TensorDataType &x) { return (x > 0) ? x : 0; });
            break;
        }

        return result;
    }

    inline NdArray backward_relu(const NdArray &input, const NdArray &grad) {
        // Shape check
        if (input.getShape() != grad.getShape()) {
            throw std::runtime_error("Invalid shape for backward_relu");
        }

        const int N = input.size();
        NdArray result(input.getShape(), input.getDevice());

        switch (input.getDevice()) {
            case Device::CPU:
                thrust::transform(thrust::host, input.getRawData(), input.getRawData() + N, grad.getRawData(),
                    result.getRawData(),
                    [] (const TensorDataType &x, const TensorDataType &g) { return (x > 0) ? g : 0; });
                break;
            case Device::GPU:
                thrust::transform(thrust::device, input.getRawData(), input.getRawData() + N, grad.getRawData(),
                    result.getRawData(),
                    [] __device__ (const TensorDataType &x, const TensorDataType &g) { return (x > 0) ? g : 0; });
            break;
        }

        return result;
    }

    inline NdArray forward_sigmoid(const NdArray &input) {
        // No shape check needed

        const int N = input.size();
        NdArray result(input.getShape(), input.getDevice());
        auto func = [] __host__ __device__ (const TensorDataType &x) { return 1.0 / (1.0 + expf(-x)); };

        switch (input.getDevice()) {
            case Device::CPU:
                thrust::transform(thrust::host, input.getRawData(), input.getRawData() + N,
                    result.getRawData(),
                    func);
            break;
            case Device::GPU:
                thrust::transform(thrust::device, input.getRawData(), input.getRawData() + N,
                    result.getRawData(),
                    func);
            break;
        }

        return result;
    }

    inline NdArray backward_sigmoid(const NdArray &input, const NdArray &grad) {
        // Shape check
        if (input.getShape() != grad.getShape()) {
            throw std::runtime_error("Invalid shape for backward_sigmoid");
        }

        const int N = input.size();
        NdArray result(input.getShape(), input.getDevice());

        switch (input.getDevice()) {
            case Device::CPU:
                thrust::transform(thrust::host, input.getRawData(), input.getRawData() + N, grad.getRawData(),
                    result.getRawData(),
                    [] (const TensorDataType &x, const TensorDataType &g) {
                        const TensorDataType s = 1.0 / (1.0 + expf(-x));
                        return s * (1 - s) * g;
                    });
                break;
            case Device::GPU:
                thrust::transform(thrust::device, input.getRawData(), input.getRawData() + N, grad.getRawData(),
                    result.getRawData(),
                    [] __device__ (const TensorDataType &x, const TensorDataType &g) {
                        const TensorDataType s = 1.0 / (1.0 + expf(-x));
                        return s * (1 - s) * g;
                    });
            break;
        }

        return result;
    }

    inline NdArray forward_fc(const NdArray &input, const NdArray &weight, const NdArray &bias) {
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

        auto [device, x, w, b] = NdArray::unifyDevice(input, weight, bias);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for forward_fc");
        }

        NdArray y({N, Cout}, Device::GPU);
        ndarray_kernel::forward_fc_kernel_gpu(x.getRawData(), w.getRawData(), b.getRawData(), y.getRawData(), N, Cin, Cout);
        return y;
    }

    inline std::tuple<NdArray, NdArray, NdArray> backward_fc(const NdArray &dy, const NdArray &input, const NdArray &weight) {
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

        auto [device, x, w, b] = NdArray::unifyDevice(input, weight, NdArray({1}, Device::CPU));

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for backward_fc");
        }

        NdArray dx({N, Cin}, Device::GPU), dw({Cout, Cin}, Device::GPU), db({Cout}, Device::GPU);
        ndarray_kernel::backward_fc_kernel_gpu(x.getRawData(), w.getRawData(), dy.getRawData(), dx.getRawData(), dw.getRawData(), db.getRawData(), N, Cin, Cout);
        return {dx, dw, db};
    }

    inline NdArray forward_max_pooling_2x2(const NdArray &input) {
        if (input.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for forward_max_pooling_2x2");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];
        const int H = input.getShape()[2];
        const int W = input.getShape()[3];

        auto [device, x] = NdArray::unifyDevice(input);

        NdArray y({N, C, H / 2, W / 2}, Device::GPU);

        cudaMemset(y.getRawData(), 0, y.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            ndarray_kernel::forward_max_pooling_2x2_kernel_gpu<<<CudaGetBlocks(C * (H / 2) * (W / 2)), kCudaThreadsNum>>>(x.getRawData() + i * C * H * W, y.getRawData() + i * C * H / 2 * W / 2, C, H, W);
        }

        return y;
    }

    inline NdArray backward_max_pooling_2x2(const NdArray & upstream_grad, const NdArray &input) {
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

        auto [device, dy, x] = NdArray::unifyDevice(upstream_grad, input);
        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for backward_max_pooling_2x2");
        }

        NdArray dx({N, C, H, W}, Device::GPU);

        cudaMemset(dx.getRawData(), 0, dx.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            ndarray_kernel::backward_max_pooling_2x2_kernel_gpu<<<CudaGetBlocks(C * (H / 2) * (W / 2)), kCudaThreadsNum>>>(
                dy.getRawData() + i * C * H / 2 * W / 2,
                x.getRawData() + i * C * H * W,
                dx.getRawData() + i * C * H * W,
                C, H, W
            );
        }

        return dx;
    }

    inline NdArray forward_softmax(const NdArray &input) {
        if (input.getShape().size() != 2) {
            throw std::runtime_error("Invalid shape for forward_softmax");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];

        auto [device, x] = NdArray::unifyDevice(input);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for forward_softmax");
        }

        NdArray y({N, C}, Device::GPU);
        ndarray_kernel::forward_softmax_kernel_gpu(x.getRawData(), y.getRawData(), N, C);

        return y;
    }

    inline NdArray cross_entropy(const NdArray &input, const NdArray &ground_truth) {
        if (input.getShape().size() != 2 || ground_truth.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for cross_entropy");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];

        auto [device, x, gt] = NdArray::unifyDevice(input, ground_truth);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for cross_entropy");
        }

        // TensorDataType *d_output_loss;
        // cudaMalloc(&d_output_loss, sizeof(TensorDataType));
        // cudaMemset(d_output_loss, 0, sizeof(TensorDataType));
        NdArray result({1}, Device::GPU);
        cudaMemset(result.getRawData(), 0, sizeof(TensorDataType));

        ndarray_kernel::cross_entropy_kernel_gpu<<<CudaGetBlocks(N), kCudaThreadsNum>>>(x.getRawData(), gt.getRawData(), result.getRawData(), N, C);

        // TensorDataType output_loss = 0.0f;
        // cudaMemcpy(&output_loss, d_output_loss, sizeof(TensorDataType), cudaMemcpyDeviceToHost);
        // cudaFree(d_output_loss);
        //
        // return output_loss;

        return result;
    }

    // The `input` argument here is the output of the softmax layer.
    inline NdArray backward_softmax_cross_entropy(const NdArray &softmax_output, const NdArray &ground_truth) {
        if (softmax_output.getShape().size() != 2 || ground_truth.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for backward_softmax_cross_entropy");
        }

        const int N = softmax_output.getShape()[0];
        const int C = softmax_output.getShape()[1];

        auto [device, x, gt] = NdArray::unifyDevice(softmax_output, ground_truth);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for backward_softmax_cross_entropy");
        }

        NdArray dx({N, C}, Device::GPU);
        ndarray_kernel::backward_softmax_cross_entropy_kernel_gpu<<<CudaGetBlocks(N * C), kCudaThreadsNum>>>(x.getRawData(), gt.getRawData(), dx.getRawData(), N, C);

        return dx;
    }

    inline NdArray conv2d(const int kernel_size, const int stride, const int padding, const NdArray &images, const NdArray &kernels) {
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

        if (K_C != C || K_H != kernel_size || K_W != kernel_size) {
            throw std::runtime_error("Invalid shape for conv2d_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size));
        }

        auto [device, im, k] = NdArray::unifyDevice(images, kernels);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for conv2d");
        }

        int H_out = (H + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W + 2 * padding - kernel_size) / stride + 1;
        NdArray y({N, K, H_out, W_out}, Device::GPU);

        TensorDataType *col;
        cudaMalloc(&col, C * kernel_size * kernel_size * H_out * W_out * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            ndarray_kernel::im2col_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                im.getRawData() + i * C * H * W, // Current image
                col,
                C, H, W, // image C, H, W
                kernel_size, kernel_size, // kernel size
                padding, padding, // padding
                stride, stride, // stride
                H_out, W_out // col H, W
            );

            cudaDeviceSynchronize();

            ndarray_kernel::gemm_row_major_gpu(
                global_cublas_handle::get_instance(), CUBLAS_OP_N, CUBLAS_OP_N,
                K, H_out * W_out, C * kernel_size * kernel_size,
                1.0f, 0.0f,
                k.getRawData(), col, y.getRawData() + i * K * H_out * W_out
            );
        }

        cudaFree(col);

        return y;
    }

    inline auto conv2d_backward(const int kernel_size, const int stride, const int padding, const NdArray& images, const NdArray& kernels, const NdArray& output_grad) {
        if (images.getShape().size() != 4 || kernels.getShape().size() != 4 || output_grad.getShape().size() != 4) {
            throw std::runtime_error("Invalid shape for conv2d backward");
        }

        const int N = images.getShape()[0];
        const int C = images.getShape()[1];
        const int H = images.getShape()[2];
        const int W = images.getShape()[3];

        const int K = kernels.getShape()[0];
        const int K_C = kernels.getShape()[1];
        const int K_H = kernels.getShape()[2];
        const int K_W = kernels.getShape()[3];

        if (K_C != C || K_H != kernel_size || K_W != kernel_size) {
            throw std::runtime_error("Invalid shape for conv2d_" + std::to_string(kernel_size) + "x" + std::to_string(kernel_size) + " backward");
        }

        int H_out = (H + 2 * padding - kernel_size) / stride + 1;
        int W_out = (W + 2 * padding - kernel_size) / stride + 1;

        if (output_grad.getShape()[0] != N || output_grad.getShape()[1] != K || output_grad.getShape()[2] != H_out || output_grad.getShape()[3] != W_out) {
            throw std::runtime_error("Output gradient shape does not match expected dimensions");
        }

        auto [device, im, k] = NdArray::unifyDevice(images, kernels);

        if (device != Device::GPU) {
            throw std::runtime_error("Unimplemented device for conv2d backward");
        }

        NdArray kernel_grad({K, C, kernel_size, kernel_size}, Device::GPU);
        TensorDataType *col;
        cudaMalloc(&col, C * kernel_size * kernel_size * H_out * W_out * sizeof(TensorDataType));

        cudaMemset(kernel_grad.getRawData(), 0, kernel_grad.size() * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            ndarray_kernel::im2col_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                im.getRawData() + i * C * H * W, // Current image
                col,
                C, H, W, // image C, H, W
                kernel_size, kernel_size, // kernel size
                padding, padding, // padding
                stride, stride, // stride
                H_out, W_out // col H, W
            );

            cudaDeviceSynchronize();

            ndarray_kernel::gemm_row_major_gpu(
                global_cublas_handle::get_instance(), CUBLAS_OP_N, CUBLAS_OP_T,
                K, C * kernel_size * kernel_size, H_out * W_out,
                1.0f, 1.0f,
                output_grad.getRawData() + i * K * H_out * W_out, col, kernel_grad.getRawData()
            );
        }

        // No need to average the kernel gradients

        cudaFree(col);

        // Compute the input gradient

        NdArray input_grad({N, C, H, W}, Device::GPU);

        TensorDataType *grad_col;
        cudaMalloc(&grad_col, C * kernel_size * kernel_size * H * W * sizeof(TensorDataType));

        for (int i = 0; i < N; i++) {
            ndarray_kernel::gemm_row_major_gpu(
                global_cublas_handle::get_instance(), CUBLAS_OP_T, CUBLAS_OP_N,
                C * kernel_size * kernel_size, H_out * W_out, K,
                1.0f, 0.0f,
                k.getRawData(), output_grad.getRawData() + i * K * H_out * W_out, grad_col
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR(("gemm_row_major_gpu" + std::to_string(i)).c_str());

            ndarray_kernel::col2im_kernel<<<CudaGetBlocks(C * H * W), kCudaThreadsNum>>>(
                grad_col,
                input_grad.getRawData() + i * C * H * W,
                C, H_out, W_out,
                kernel_size, kernel_size,
                padding, padding,
                stride, stride,
                H, W
            );
            cudaDeviceSynchronize();
            CHECK_CUDA_ERROR("col2im_kernel");
        }
        cudaFree(grad_col);

        return std::tuple{input_grad, kernel_grad};
    }
}

#endif //NDARRAYNN_CUH
