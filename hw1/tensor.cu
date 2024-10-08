//
// Created by timetraveler314 on 9/22/24.
//

#include "tensor.h"

Tensor::Tensor(std::vector<int> shape, const TensorDevice device): device(device), shape(std::move(shape)) {
    int bufferSize = 1;
    for (const auto dim: this->shape) {
        bufferSize *= dim;
    }

    this->data = device_ptr(device, bufferSize);
}

Tensor::Tensor(const Tensor &tensor) : device(tensor.device), shape(tensor.shape), data(tensor.data) {}

Tensor Tensor::gpu() const {
    Tensor gpuTensor(shape, TensorDevice::GPU);
    gpuTensor.data = data.copy_to(TensorDevice::GPU);
    return gpuTensor;
}

Tensor Tensor::cpu() const {
    Tensor cpuTensor(shape, TensorDevice::CPU);
    cpuTensor.data = data.copy_to(TensorDevice::CPU);
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
                os << std::fixed << std::setprecision(8) << data->space[offset + i];
            } else {
                os << std::fixed << std::setprecision(8) << data->space[offset + i] << ", ";
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

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    tensor.print(os);
    return os;
}

Tensor operator+(const Tensor &t1, const Tensor &t2) {

}

Tensor Tensor::relu() const {
    Tensor result(shape, device);
    switch (device) {
        case TensorDevice::CPU: {
            for (int i = 0; i < size(); i++) {
                result.data->space[i] = data->space[i] > 0 ? data->space[i] : 0;
            }
            return result;
        }
        case TensorDevice::GPU: {
            TensorKernel::relu_gpu<<<CudaGetBlocks(size()), kCudaThreadsNum>>>(data->space, result.data->space, size());
            cudaDeviceSynchronize();
            return result;
        }
    }
}

Tensor Tensor::sigmoid() const {
    Tensor result(shape, device);
    switch (device) {
        case TensorDevice::CPU: {
            for (int i = 0; i < size(); i++) {
                result.data->space[i] = 1.0 / (1.0 + exp(-data->space[i]));
            }
            return result;
        }
        case TensorDevice::GPU: {
            TensorKernel::sigmoid_gpu<<<CudaGetBlocks(size()), kCudaThreadsNum>>>(data->space, result.data->space, size());
            cudaDeviceSynchronize();
            return result;
        }
    }
}

__global__ void TensorKernel::relu_gpu(const TensorDataType* in, TensorDataType* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void TensorKernel::sigmoid_gpu(const TensorDataType *in, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = 1.0 / (1.0 + expf(-in[i]));
    }
}
