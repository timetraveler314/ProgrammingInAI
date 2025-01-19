//
// Created by timetraveler314 on 10/25/24.
//
#pragma once
#ifndef NDARRAY_KERNEL_H
#define NDARRAY_KERNEL_H

#include "ndarray.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

void checkCudaError(const char* msg);

#define CHECK_CUDA_ERROR(msg) checkCudaError(msg)

// Use 512 or 256 threads per block
constexpr int kCudaThreadsNum = 512;
int CudaGetBlocks(const int N);

// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); \
    i += blockDim.x * gridDim.x)

namespace ndarray_kernel {
    __global__ void ewise_add_kernel_gpu(const TensorDataType* a, const TensorDataType* b, TensorDataType* c, size_t size);

    __global__ void ewise_minus_kernel_gpu(const TensorDataType* a, const TensorDataType* b, TensorDataType* c, size_t size);

    void gemm_row_major_gpu(const cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
                                   const int m, const int n, const int k, const TensorDataType alpha, const TensorDataType beta,
                                   const TensorDataType* A, const TensorDataType* B, TensorDataType* C);

    __global__ void ones_kernel(TensorDataType* data, size_t size);

    // Computes the forward pass of a fully connected layer using cuBLAS gemm.
    //
    // Parameters:
    //   input: Tensor of shape [batch_size, input_size]
    //   weight: Tensor of shape [output_size, input_size]
    //   bias: Tensor of shape [output_size], broadcast to [batch_size, output_size]
    //
    // Outputs:
    //   output: Tensor of shape [batch_size, output_size], computed as:
    //   output = input * weight^T + bias
    void forward_fc_kernel_gpu(const TensorDataType* input, const TensorDataType* weight, const TensorDataType* bias, TensorDataType* output, size_t batch_size, size_t input_size, size_t output_size);

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
    void backward_fc_kernel_gpu(const TensorDataType* input, const TensorDataType* weight, const TensorDataType* output_grad,
        TensorDataType* input_grad, TensorDataType* weight_grad, TensorDataType* bias_grad,
        size_t batch_size, size_t input_size, size_t output_size);

    void forward_softmax_kernel_gpu(TensorDataType *input, TensorDataType *output, size_t batch_size, size_t num_classes);

    __global__ void cross_entropy_kernel_gpu(TensorDataType *input, TensorDataType *target, TensorDataType * output_loss, size_t batch_size, size_t num_classes);

    __global__ void backward_softmax_cross_entropy_kernel_gpu(TensorDataType *softmax_output, TensorDataType *target, TensorDataType *output_grad, size_t batch_size, size_t num_classes);

    __global__ void forward_max_pooling_2x2_kernel_gpu(const TensorDataType *input, TensorDataType *output, size_t num_channels, size_t height, size_t width);

    __global__ void backward_max_pooling_2x2_kernel_gpu(const TensorDataType *upstream_grad, const TensorDataType *input, TensorDataType *output_grad, size_t num_channels, size_t height, size_t width);

    __global__ void im2col_kernel(const float* input, float* output,
                                  int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w,
                                  int out_h, int out_w);

    __global__ void col2im_kernel(const float* col, float* im,
                                  int C, int H, int W,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w,
                                  int out_h, int out_w);

    __global__ void average_dim_1_kernel(const float* input, float* output, int N, int D);
};



#endif //TENSOR_KERNEL_H
