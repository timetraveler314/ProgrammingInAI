//
// Created by timetraveler314 on 9/22/24.
//

#include "ndarray.h"

#include "../global_curand_generator.cuh"

NdArray::NdArray(std::vector<int> shape, const Device device): device(device), shape(std::move(shape)) {
    int bufferSize = 1;
    for (const auto dim: this->shape) {
        bufferSize *= dim;
    }

    this->data = device_ptr(device, bufferSize);
}

NdArray::NdArray(const NdArray &tensor) : device(tensor.device), shape(tensor.shape), data(tensor.data) {}

NdArray NdArray::ones(std::vector<int> shape, Device device) {
    NdArray result(shape, Device::CPU);
    for (int i = 0; i < result.size(); i++) {
        result.data->space[i] = 1.0;
    }

    switch (device) {
        case Device::CPU:
            return result;
        case Device::GPU:
            return result.gpu();
    }
    return result;
}

NdArray NdArray::iota(std::vector<int> shape, Device device) {
    NdArray result(shape, Device::CPU);
    for (int i = 0; i < result.size(); i++) {
        result.data->space[i] = i;
    }
    return device == Device::CPU ? result : result.gpu();
}

NdArray NdArray::uniform(std::vector<int> shape, Device device) {
    constexpr TensorDataType low = 0.0f, high = 1.0f;
    if (device == Device::CPU) {
        NdArray resultCPU(shape, Device::CPU);
        for (int i = 0; i < resultCPU.size(); i++) {
            resultCPU.getRawData()[i] = low + static_cast<TensorDataType>(rand()) / RAND_MAX * (high - low);
        }
        return resultCPU;
    } else {
        NdArray resultGPU(shape, Device::GPU);
        // Use cuRAND
        // curandGenerateUniform(global_curand_generator::get_instance(), resultGPU.getRawData(), resultGPU.size());
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, rand());
        curandGenerateUniform(gen, resultGPU.getRawData(), resultGPU.size());
        return resultGPU;
    }
}

NdArray NdArray::view(const std::vector<int> &newShape) const {
    NdArray newTensor(newShape, device);
    newTensor.data = data;
    return newTensor;
}

NdArray NdArray::gpu() const {
    NdArray gpuTensor(shape, Device::GPU);
    gpuTensor.data = data.copy_to(Device::GPU);
    return gpuTensor;
}

Device NdArray::getDevice() const {
    return device;
}

TensorDataType * NdArray::getRawData() const {
    return data->space;
}

NdArray NdArray::cpu() const {
    NdArray cpuTensor(shape, Device::CPU);
    cpuTensor.data = data.copy_to(Device::CPU);
    return cpuTensor;
}

int NdArray::size() const {
    int bufferSize = 1;
    for (const auto dim : shape) {
        bufferSize *= dim;
    }
    return bufferSize;
}

void NdArray::print(std::ostream &os, const int depth, const int offset) const {
    if (device == Device::GPU) {
        NdArray cpuTensor = cpu();
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

std::string NdArray::toString() const {
    std::stringstream ss;
    print(ss);
    return ss.str();
}

std::vector<int> NdArray::getShape() const {
    return this->shape;
}

std::ostream &operator<<(std::ostream &os, const NdArray &tensor) {
    tensor.print(os);
    return os;
}
