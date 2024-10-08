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

Tensor::Tensor(const Tensor &tensor) : device(tensor.device), shape(tensor.shape) {
    int bufferSize = 1;
    for (const auto dim : shape) {
        bufferSize *= dim;
    }

    switch (device) {
        case TensorDevice::CPU:
            data = new TensorDataType[bufferSize];
            memcpy(data, tensor.data, bufferSize * sizeof(TensorDataType));
            break;
        case TensorDevice::GPU:
            cudaMalloc(&data, bufferSize * sizeof(TensorDataType));
            cudaMemcpy(data, tensor.data, bufferSize * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
            break;
    }
}

Tensor::Tensor(Tensor &&tensor) : device(tensor.device), shape(std::move(tensor.shape)), data(tensor.data) {
    tensor.data = nullptr;
}

Tensor::~Tensor() {
    if (data) {
        switch (device) {
            case TensorDevice::CPU:
                delete[] data;
            break;
            case TensorDevice::GPU:
                cudaFree(data);
            break;
        }
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
                os << std::fixed << std::setprecision(8) << data[offset + i];
            } else {
                os << std::fixed << std::setprecision(8) << data[offset + i] << ", ";
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
    if (t1.device == TensorDevice::GPU || t2.device == TensorDevice::GPU) {

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
                result.data[i] = 1.0 / (1.0 + exp(-data[i]));
            }
            return result;
        }
        case TensorDevice::GPU: {
            TensorKernel::sigmoid_gpu<<<CudaGetBlocks(size()), kCudaThreadsNum>>>(data, result.data, size());
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
