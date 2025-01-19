//
// Created by timetraveler314 on 9/22/24.
//

#include "ndarray.h"

#include <ndarray_kernel.cuh>
#include <nn.cuh>

#include "../global_curand_generator.cuh"
#include "../utils/global_cublas_handle.cuh"

NdArray::NdArray(std::vector<int> shape, const Device device): device(device), shape(std::move(shape)) {
    int bufferSize = 1;
    for (const auto dim: this->shape) {
        bufferSize *= dim;
    }

    this->data = device_ptr(device, bufferSize);
}

NdArray::NdArray(const NdArray &tensor) : device(tensor.device), shape(tensor.shape), data(tensor.data) {}

NdArray NdArray::zeros_like(const NdArray &nds) {
    NdArray result(nds.getShape(), Device::CPU);

    for (int i = 0; i < result.size(); i++) {
        result.data->space[i] = 0.0;
    }

    if (nds.getDevice() == Device::GPU) {
        return result.gpu();
    }
    return result;
}

NdArray NdArray::zeros(std::vector<int> shape, Device device) {
    NdArray result(shape, Device::CPU);
    for (int i = 0; i < result.size(); i++) {
        result.data->space[i] = 0.0;
    }

    switch (device) {
        case Device::CPU:
            return result;
        case Device::GPU:
            return result.gpu();
    }
    return result;
}

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

/* xaiver - Xavier initialization
 *
 * @param shape: shape [m, n] of the tensor, only >=2D tensors are supported
 * @param device: device to store the tensor, only GPU tensors are supported
 *
 * @description: the tensor is initialized with random values drawn from a normal distribution with mean 0 and variance 2 / (m + n)
 *
 * @return: tensor with Xavier initialization
 */
NdArray NdArray::xavier(const std::vector<int> &shape, const Device device) {
    if (shape.size() < 2) {
        throw std::runtime_error("Xavier initialization is only supported for >=2D tensors");
    }

    if (device != Device::GPU) {
        throw std::runtime_error("Xavier initialization is only supported for GPU tensors");
    }

    const int m = shape[0];
    const int n = shape[1];
    const TensorDataType variance = 2.0f / (m + n);

    // Use curand
    NdArray result(shape, Device::GPU);
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(gen, rand());
    curandGenerateNormal(gen, result.getRawData(), result.size(), 0.0f, sqrt(variance));
    curandDestroyGenerator(gen);
    return result;
}

NdArray NdArray::from_raw_data(std::vector<int> shape, Device device, TensorDataType *data) {
    NdArray result(shape, device);
    if (device == Device::CPU) {
        for (int i = 0; i < result.size(); i++) {
            result.getRawData()[i] = data[i];
        }
    } else {
        cudaMemcpy(result.getRawData(), data, result.size() * sizeof(TensorDataType), cudaMemcpyHostToDevice);
    }
    return result;
}

NdArray NdArray::view(const std::vector<int> &newShape) const {
    NdArray newTensor(newShape, device);
    newTensor.data = data;
    return newTensor;
}

NdArray NdArray::reshape(const std::vector<int> &newShape) const {
    NdArray newTensor(newShape, device);
    if (size() != newTensor.size()) throw std::runtime_error("Reshape size mismatch");

    if (device == Device::CPU) {
        for (int i = 0; i < size(); i++) {
            newTensor.getRawData()[i] = getRawData()[i];
        }
    } else {
        cudaMemcpy(newTensor.getRawData(), getRawData(), size() * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
    }

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

NdArray NdArray::operator-() const {
    if (device == Device::CPU) {
        NdArray result(shape, Device::CPU);
        for (int i = 0; i < size(); i++) {
            result.getRawData()[i] = -getRawData()[i];
        }
        return result;
    } else {
        NdArray result(shape, Device::GPU);
        thrust::transform(thrust::device, getRawData(), getRawData() + size(), result.getRawData(), thrust::negate<TensorDataType>());
        return result;
    }
}

NdArray NdArray::transpose() const {
    if (shape.size() != 2) {
        throw std::runtime_error("Transpose is only supported for 2D tensors");
    }

    NdArray result({shape[1], shape[0]}, device);
    if (device == Device::CPU) {
        for (int i = 0; i < shape[0]; i++) {
            for (int j = 0; j < shape[1]; j++) {
                result.getRawData()[j * shape[0] + i] = getRawData()[i * shape[1] + j];
            }
        }
    } else {
        // Use Sgeam to transpose the matrix
        const TensorDataType alpha = 1.0f, beta = 0.0f;
        cublasSgeam(global_cublas_handle::get_instance(), CUBLAS_OP_T, CUBLAS_OP_T, shape[0], shape[1], &alpha, getRawData(), shape[1], &beta, getRawData(), shape[1], result.getRawData(), shape[0]);
    }
    return result;
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

NdArray operator*(TensorDataType scalar, const NdArray &tensor) {
    NdArray result(tensor.getShape(), tensor.getDevice());

    switch (tensor.getDevice()) {
        case Device::CPU:
            for (int i = 0; i < tensor.size(); i++) {
                result.getRawData()[i] = scalar * tensor.getRawData()[i];
            }
            break;
        case Device::GPU:
            thrust::transform(thrust::device, tensor.getRawData(), tensor.getRawData() + tensor.size(), result.getRawData(), [scalar] __device__(TensorDataType x) { return scalar * x; });
            break;
    }

    return result;
}

/*
 * operator% - Matrix multiplication
 *
 * @param lhs: m x k matrix
 *        rhs: k x n matrix
 *
 * @return m x n matrix
 */
NdArray operator%(const NdArray &lhs, const NdArray &rhs) {
    auto [device, x, y] = NdArray::unifyDevice(lhs, rhs);

    int m = x.getShape()[0];
    int k = x.getShape()[1];
    int n = y.getShape()[1];

    if (x.getShape()[1] != y.getShape()[0]) throw std::runtime_error("Shape mismatch in matrix multiplication");

    if (device == Device::CPU) {
        NdArray result({m, n}, Device::CPU);
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                result.getRawData()[i * n + j] = 0;
                for (int l = 0; l < k; l++) {
                    result.getRawData()[i * n + j] += x.getRawData()[i * k + l] * y.getRawData()[l * n + j];
                }
            }
        }
        return result;
    } else {
        NdArray result({m, n}, Device::GPU);
        ndarray_kernel::gemm_row_major_gpu(global_cublas_handle::get_instance(), CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, 1.0f, 0.0f, x.getRawData(), y.getRawData(), result.getRawData());
        return result;
    }
}

std::ostream &operator<<(std::ostream &os, const NdArray &tensor) {
    tensor.print(os);
    return os;
}

NdArray operator+(const NdArray &lhs, const NdArray &rhs) {
    auto [device, x, y] = NdArray::unifyDevice(lhs, rhs);

    if (x.getShape() != y.getShape()) {
        throw std::runtime_error("Shape mismatch in NdArray addition");
    }

    if (device == Device::CPU) {
        NdArray result(lhs.getShape(), Device::CPU);
        for (int i = 0; i < result.size(); i++) {
            result.getRawData()[i] = x.getRawData()[i] + y.getRawData()[i];
        }
        return result;
    } else {
        NdArray result(lhs.getShape(), Device::GPU);
        std::cout << "Calling ewise_add_kernel_gpu" << std::endl;
        ndarray_kernel::ewise_add_kernel_gpu<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(x.getRawData(), y.getRawData(), result.getRawData(), result.size());

        return result;
    }
}

NdArray operator-(const NdArray &lhs, const NdArray &rhs) {
    auto [device, x, y] = NdArray::unifyDevice(lhs, rhs);

    if (x.getShape() != y.getShape()) {
        throw std::runtime_error("Shape mismatch in NdArray operator-");
    }

    if (device == Device::CPU) {
        NdArray result(lhs.getShape(), Device::CPU);
        for (int i = 0; i < result.size(); i++) {
            result.getRawData()[i] = x.getRawData()[i] + y.getRawData()[i];
        }
        return result;
    } else {
        NdArray result(lhs.getShape(), Device::GPU);
        std::cout << "Calling ewise_minus_kernel_gpu" << std::endl;
        ndarray_kernel::ewise_minus_kernel_gpu<<<CudaGetBlocks(result.size()), kCudaThreadsNum>>>(x.getRawData(), y.getRawData(), result.getRawData(), result.size());

        return result;
    }
}