//
// Created by timetraveler314 on 9/22/24.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <iostream>

// Use 512 or 256 threads per block
const int kCudaThreadsNum = 512;
inline int CudaGetBlocks(const int N) {
    return (N + kCudaThreadsNum - 1) / kCudaThreadsNum;
}
// Define the grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
    i < (n); \
    i += blockDim.x * gridDim.x)

typedef double TensorDataType;

enum class TensorDevice {
    CPU,
    GPU
};

class Tensor {
public:
    TensorDevice device;
    std::vector<int> shape;
    TensorDataType* data;

public:
    Tensor(std::vector<int> shape, TensorDevice device);
    ~Tensor();

    Tensor cpu() const;
    Tensor gpu() const;

    int size() const;

    void print(std::ostream& os, int depth = 0, int offset = 0) const;

    Tensor relu() const;
};

namespace TensorKernel {
    __global__ void relu_gpu(const TensorDataType* in, TensorDataType* out, int size);
}



#endif //TENSOR_H
