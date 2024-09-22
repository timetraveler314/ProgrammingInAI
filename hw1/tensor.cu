//
// Created by timetraveler314 on 9/22/24.
//

#include "tensor.h"

Tensor::Tensor(std::vector<int> shape, const TensorDevice device): device(device), shape(std::move(shape)) {
    int bufferSize = 1;
    for (const auto dim : this->shape) {
        bufferSize *= dim;
    }

    switch (device) {
        case TensorDevice::CPU:
            data = new TensorDataType[bufferSize];
            break;
        case TensorDevice::GPU:
            cudaMalloc(&data, bufferSize * sizeof(TensorDataType));
            break;
    }
}

Tensor::~Tensor() {
    switch (device) {
        case TensorDevice::CPU:
            delete[] data;
            break;
        case TensorDevice::GPU:
            cudaFree(data);
            break;
    }
}

Tensor Tensor::gpu() const {
    Tensor gpuTensor(shape, TensorDevice::GPU);
    switch (device) {
        case TensorDevice::CPU:
            cudaMemcpy(gpuTensor.data, data, size() * sizeof(TensorDataType), cudaMemcpyHostToDevice);
        case TensorDevice::GPU:
            cudaMemcpy(gpuTensor.data, data, size() * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
    }

    return gpuTensor;
}

Tensor Tensor::cpu() const {
    Tensor cpuTensor(shape, TensorDevice::CPU);
    switch (device) {
        case TensorDevice::CPU:
            memcpy(cpuTensor.data, data, size() * sizeof(TensorDataType));
        case TensorDevice::GPU:
            cudaMemcpy(cpuTensor.data, data, size() * sizeof(TensorDataType), cudaMemcpyDeviceToHost);
    }
    return cpuTensor;
}


int Tensor::size() const {
    int bufferSize = 1;
    for (const auto dim : shape) {
        bufferSize *= dim;
    }
    return bufferSize;
}

void Tensor::print(std::ostream &os, int depth, int offset) const {
    if (device == TensorDevice::GPU) {
        Tensor cpuTensor = cpu();
        cpuTensor.print(os, depth, offset);
        return;
    }

    int stride = 1;
    for (int i = depth + 1; i < shape.size(); i++) {
        stride *= shape[i];
    }

    if (depth + 1 == shape.size()) {
        os << "[";
        for (int i = 0; i < shape[depth]; i++) {
            if (i == shape[depth] - 1) {
                os << data[offset + i];
            } else {
                os << data[offset + i] << ", ";
            }
        }
        os << "]";
    } else {
        os << "[";
        for (int i = 0; i < shape[depth]; i++) {
            print(os, depth + 1, stride * i + offset);
            if (i != shape[depth] - 1) {
                os << ",\n";
            }
        }
        os << "]";
    }
}

Tensor Tensor::relu() const {
    Tensor result(shape, device);
    switch (device) {
        case TensorDevice::CPU: {
            for (int i = 0; i < size(); i++) {
                result.data[i] = data[i] > 0 ? data[i] : 0;
            }
            return result;
        }
        case TensorDevice::GPU: {
            TensorKernel::relu_gpu<<<CudaGetBlocks(size()), kCudaThreadsNum>>>(data, result.data, size());
            return result;
        }
    }
}

__global__ void TensorKernel::relu_gpu(const TensorDataType* in, TensorDataType* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}
