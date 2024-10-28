//
// Created by timetraveler314 on 9/22/24.
//

#include "tensor.h"

#include "global_curand_generator.cuh"

Tensor::Tensor(std::vector<int> shape, const TensorDevice device): device(device), shape(std::move(shape)) {
    int bufferSize = 1;
    for (const auto dim: this->shape) {
        bufferSize *= dim;
    }

    this->data = device_ptr(device, bufferSize);
}

Tensor::Tensor(const Tensor &tensor) : device(tensor.device), shape(tensor.shape), data(tensor.data) {}

Tensor Tensor::ones(std::vector<int> shape, TensorDevice device) {
    Tensor result(shape, TensorDevice::CPU);
    for (int i = 0; i < result.size(); i++) {
        result.data->space[i] = 1.0;
    }

    switch (device) {
        case TensorDevice::CPU:
            return result;
        case TensorDevice::GPU:
            return result.gpu();
    }
    return result;
}

Tensor Tensor::uniform(std::vector<int> shape, TensorDevice device, TensorDataType low, TensorDataType high) {
    if (device == TensorDevice::CPU) {
        Tensor resultCPU(shape, TensorDevice::CPU);
        for (int i = 0; i < resultCPU.size(); i++) {
            resultCPU.getRawData()[i] = low + static_cast<TensorDataType>(rand()) / RAND_MAX * (high - low);
        }
        return resultCPU;
    } else {
        Tensor resultGPU(shape, TensorDevice::GPU);
        // Use cuRAND
        curandGenerateUniform(global_curand_generator::get_instance(), resultGPU.getRawData(), resultGPU.size());
        return resultGPU;
    }
}

Tensor Tensor::gpu() const {
    Tensor gpuTensor(shape, TensorDevice::GPU);
    gpuTensor.data = data.copy_to(TensorDevice::GPU);
    return gpuTensor;
}

TensorDataType * Tensor::getRawData() const {
    return data->space;
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

void Tensor::print(std::ostream &os, const int depth, const int offset) const {
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

std::vector<int> Tensor::getShape() const {
    return this->shape;
}

std::ostream &operator<<(std::ostream &os, const Tensor &tensor) {
    tensor.print(os);
    return os;
}
