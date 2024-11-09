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
};



#endif //TENSOR_KERNEL_H
