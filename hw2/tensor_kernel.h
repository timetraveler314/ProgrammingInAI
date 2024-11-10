//
// Created by timetraveler314 on 10/25/24.
//

#ifndef TENSOR_KERNEL_H
#define TENSOR_KERNEL_H

#include "tensor.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda/std/span>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

// Use 512 or 256 threads per block
constexpr int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); \
    i += blockDim.x * gridDim.x)

namespace tensor_kernel {
    inline void gemm_row_major_gpu(const cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                        const int m, const int n, const int k, const TensorDataType alpha, const TensorDataType beta,
                        const TensorDataType* A, const TensorDataType* B, TensorDataType* C) {
        TensorDataType alpha_local = 1.0;
        TensorDataType beta_local = 0.0;

        TensorDataType* tmp = nullptr;
        cudaMalloc(&tmp, m * n * sizeof(TensorDataType));
        cudaMemcpy(tmp, C, m * n * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);

        // Transpose C to allow for beta != 0
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
            m, n,
            &alpha_local,
            C, n,
            &beta_local,
            C, n,
            tmp, m);

        alpha_local = alpha;
        beta_local = beta;
        cublasSgemm(handle, (transa == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N, (transb == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m, n, k,
                    &alpha_local,
                    A, (transa == CUBLAS_OP_N) ? k : m,
                    B, (transb == CUBLAS_OP_N) ? n : k,
                    &beta_local,
                    tmp, m);

        // Transpose the result
        alpha_local = 1.0;
        beta_local = 0.0;
        cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_T,
                    n, m,
                    &alpha_local,
                    tmp, m,
                    &beta_local,
                    tmp, m,
                    C, n);

        // The original code: (which doesn't work)
        // cublasSgemm(handle, (transa == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N, (transb == CUBLAS_OP_N) ? CUBLAS_OP_T : CUBLAS_OP_N,
        //             n, m, k,
        //             &alpha_local,
        //             B, (transb == CUBLAS_OP_N) ? n : k,
        //             A, (transa == CUBLAS_OP_N) ? k : m,
        //             &beta_local,
        //             C, n);
    }

    __global__ void ones_kernel(TensorDataType* data, size_t size) {
        CUDA_KERNEL_LOOP(i, size) {
            data[i] = 1.0;
        }
    }

    // Computes the forward pass of a fully connected layer using cuBLAS gemm.
    //
    // Parameters:
    //   input: Tensor of shape [batch_size, input_size]
    //   weight: Tensor of shape [output_size, input_size]
    //   bias: Tensor of shape [output_size], broadcasted to [batch_size, output_size]
    //
    // Outputs:
    //   output: Tensor of shape [batch_size, output_size], computed as:
    //   output = input * weight^T + bias
    inline void forward_fc_kernel_gpu(const TensorDataType* input, const TensorDataType* weight, const TensorDataType* bias, TensorDataType* output, size_t batch_size, size_t input_size, size_t output_size) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Compute the matrix multiplication
        gemm_row_major_gpu(handle, CUBLAS_OP_N, CUBLAS_OP_T,
            batch_size, output_size, input_size,
            1.0, 0.0,
            input, weight, output);

        TensorDataType* ones = nullptr;
        cudaMalloc(&ones, batch_size * sizeof(TensorDataType));
        ones_kernel<<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(ones, batch_size);

        cudaDeviceSynchronize();

        // Broadcast the bias
        gemm_row_major_gpu(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            batch_size, output_size, 1,
            1.0, 1.0,
            ones, bias, output);

        cudaDeviceSynchronize();
        cublasDestroy(handle);
        cudaFree(ones);
    }

    // Use cuBLAS gemm to compute the backward pass of a fully connected layer
    // input: [batch_size, input_size]
    // weight: [output_size, input_size]
    // output_grad: [batch_size, output_size]

    // input_grad = output_grad * weight
    // input_grad: [batch_size, input_size]

    // weight_grad = output_grad^T * input
    // weight_grad: [output_size, input_size]

    // bias_grad = output_grad^T * ones
    // bias_grad: [output_size]
    inline void backward_fc_kernel_gpu(const TensorDataType* input, const TensorDataType* weight, const TensorDataType* output_grad,
        TensorDataType* input_grad, TensorDataType* weight_grad, TensorDataType* bias_grad,
        size_t batch_size, size_t input_size, size_t output_size) {
        cublasHandle_t handle;
        cublasCreate(&handle);

        // Compute the input gradient
        gemm_row_major_gpu(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            batch_size, input_size, output_size,

            1.0, 0.0,
            output_grad, weight, input_grad);

        // Compute the weight gradient
        gemm_row_major_gpu(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            output_size, input_size, batch_size,
            1.0, 0.0,
            output_grad, input, weight_grad);

        TensorDataType* ones = nullptr;
        cudaMalloc(&ones, batch_size * sizeof(TensorDataType));
        ones_kernel<<<CudaGetBlocks(batch_size), kCudaThreadsNum>>>(ones, batch_size);

        // Compute the bias gradient
        gemm_row_major_gpu(handle, CUBLAS_OP_T, CUBLAS_OP_N,
            output_size, 1, batch_size,
            1.0, 0.0,
            output_grad, ones, bias_grad);

        cudaDeviceSynchronize();
        cublasDestroy(handle);
        cudaFree(ones);
    }

    inline void forward_softmax_kernel_gpu(TensorDataType *input, TensorDataType *output, size_t batch_size, size_t num_classes) {
        for (int i = 0; i < batch_size; i++) {
            auto current_row = input + i * num_classes;
            auto current_output = output + i * num_classes;
            auto max_val = thrust::reduce(thrust::device, current_row, current_row + num_classes,
                std::numeric_limits<TensorDataType>::min(), thrust::maximum<TensorDataType>());
            thrust::transform(thrust::device, current_row, current_row + num_classes, current_output,
                [max_val] __device__ (TensorDataType val) { return exp(val - max_val); });

            auto normalizer = thrust::reduce(thrust::device, current_output, current_output + num_classes, 0.0f, thrust::plus<TensorDataType>());
            thrust::transform(thrust::device, current_output, current_output + num_classes, current_output,
                [normalizer] __device__ (TensorDataType val) { return val / normalizer; });
        }
    }

    __global__ void cross_entropy_kernel_gpu(TensorDataType *input, TensorDataType *target, TensorDataType * output_loss, size_t batch_size, size_t num_classes) {
        CUDA_KERNEL_LOOP(i, batch_size) {
            int label = static_cast<int>(target[i]);
            TensorDataType my_loss = 0.0;
            if (label >= 0 && label < num_classes) {
                my_loss = -log(input[i * num_classes + label] + 1e-8);
            }

            atomicAdd(output_loss, my_loss / batch_size);
        }
    }
    __global__ void backward_softmax_cross_entropy_kernel_gpu(TensorDataType *softmax_input, TensorDataType *target, TensorDataType *output_grad, size_t batch_size, size_t num_classes) {
        CUDA_KERNEL_LOOP(i, batch_size * num_classes) {
            int sample_idx = i / num_classes;
            int class_idx = i % num_classes;

            output_grad[i] = softmax_input[i] - (target[sample_idx] == class_idx);
        }
    }

    __global__ void forward_max_pooling_2x2_kernel_gpu(const TensorDataType *input, TensorDataType *output, size_t num_channels, size_t height, size_t width) {
        CUDA_KERNEL_LOOP(index, num_channels * height / 2 * width / 2) {
            int w = index % (width / 2);
            int h = (index / (width / 2)) % (height / 2);
            int c = (index / (width / 2) / (height / 2)) % num_channels;
            // int n = index / (width / 2) / (height / 2) / num_channels;

            const int input_idx = c * height * width + h * 2 * width + w * 2;
            TensorDataType max_val = input[input_idx];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    max_val = fmaxf(max_val, input[input_idx + i * width + j]);
                }
            }

            output[index] = max_val;
        }
    }

    __global__ void backward_max_pooling_2x2_kernel_gpu(const TensorDataType *upstream_grad, const TensorDataType *input, TensorDataType *output_grad, size_t num_channels, size_t height, size_t width) {
        CUDA_KERNEL_LOOP(index, num_channels * height / 2 * width / 2) {
            int w = index % (width / 2);
            int h = (index / (width / 2)) % (height / 2);
            int c = (index / (width / 2) / (height / 2)) % num_channels;

            const int output_idx = c * height / 2 * width / 2 + h * width / 2 + w;
            const int input_idx = c * height * width + h * 2 * width + w * 2;

            TensorDataType max_val = input[input_idx];
            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    max_val = fmaxf(max_val, input[input_idx + i * width + j]);
                }
            }

            for (int i = 0; i < 2; i++) {
                for (int j = 0; j < 2; j++) {
                    output_grad[input_idx + i * width + j] = (input[input_idx + i * width + j] == max_val) * upstream_grad[output_idx];
                }
            }
        }
    }

    __global__ void im2col_kernel(const float* input, float* output,
                                  int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w,
                                  int out_h, int out_w) {
        CUDA_KERNEL_LOOP(col_index, C * out_h * out_w) {
            int w_out = col_index % out_w;
            int h_out = (col_index / out_w) % out_h;
            int c_in = (col_index / (out_w * out_h)) % C;

            int h_in_start = h_out * stride_h - pad_h;
            int w_in_start = w_out * stride_w - pad_w;

            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_in = h_in_start + kh;
                    int w_in = w_in_start + kw;
                    int output_index = ((c_in * kernel_h * kernel_w + kh * kernel_w + kw) * out_h + h_out) * out_w + w_out;

                    if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                        int input_index = (c_in * H + h_in) * W + w_in;
                        output[output_index] = input[input_index];
                    } else {
                        output[output_index] = 0.0f;  // Zero-padding
                    }
                }
            }
        }
    }

    __global__ void col2im_kernel(const float* col, float* im,
                                  int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w,
                                  int out_h, int out_w) {
        CUDA_KERNEL_LOOP(im_index, C * H * W) {
            int w = im_index % W;
            int h = (im_index / W) % H;
            int c = im_index / (W * H);

            float val = 0.0f;
            int count = 0; // Number of column elements contributing to this output element
            for (int kh = 0; kh < kernel_h; ++kh) {
                for (int kw = 0; kw < kernel_w; ++kw) {
                    int h_out = (h + pad_h - kh) / stride_h;
                    int w_out = (w + pad_w - kw) / stride_w;
                    if (h_out >= 0 && h_out < out_h && w_out >= 0 && w_out < out_w) {
                        int col_index = ((c * kernel_h * kernel_w + kh * kernel_w + kw) * out_h + h_out) * out_w + w_out;
                        val += col[col_index];
                        count++;
                    }
                }
            }
            im[im_index] = val / count;
        }
    }
};



#endif //TENSOR_KERNEL_H
