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

Tensor Tensor::gpu() const {
    Tensor gpuTensor(shape, TensorDevice::GPU);
    gpuTensor.data = data.copy_to(TensorDevice::GPU);
    return gpuTensor;
}

std::tuple<TensorDevice, Tensor, Tensor> Tensor::unifyDevice(const Tensor &lhs, const Tensor &rhs) {
    if (lhs.device == TensorDevice::GPU || rhs.device == TensorDevice::GPU) {
        switch (lhs.device) {
            case TensorDevice::CPU: {
                Tensor lhsGpu = lhs.gpu();
                return {TensorDevice::GPU, lhsGpu, rhs};
            }
            case TensorDevice::GPU: {
                switch (rhs.device) {
                    case TensorDevice::CPU: {
                        Tensor rhsGpu = rhs.gpu();
                        return {TensorDevice::GPU, lhs, rhsGpu};
                    }
                    case TensorDevice::GPU: {
                        return {TensorDevice::GPU, lhs, rhs};
                    }
                }
            }
        }
    } else {
        return {TensorDevice::CPU, lhs, rhs};
    }

    throw std::runtime_error("Unreachable code");
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

void Tensor::acceptModifier(const std::function<void(DeviceSpace &)> &modifier) const {
    modifier(*data);
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

Tensor operator+(const Tensor &t1, const Tensor &t2) {
    if (t1.shape != t2.shape) {
        throw std::runtime_error("Shapes do not match");
    }

    auto [device_result, t1_unified, t2_unified] = Tensor::unifyDevice(t1, t2);

    Tensor result = Tensor(t1_unified.shape, device_result);

    switch (device_result) {
        case TensorDevice::CPU:
            for (int i = 0; i < t1_unified.size(); i++) {
                result.data->space[i] = t1_unified.data->space[i] + t2_unified.data->space[i];
            }
            break;
        case TensorDevice::GPU:
            TensorKernel::add_gpu<<<CudaGetBlocks(t1_unified.size()), kCudaThreadsNum>>>(t1_unified.data->space, t2_unified.data->space, result.data->space, t1_unified.size());
            cudaDeviceSynchronize();
            break;
    }

    return result;
}

Tensor operator-(const Tensor &t1, const Tensor &t2) {
    if (t1.shape != t2.shape) {
        throw std::runtime_error("Shapes do not match");
    }

    auto [device_result, t1_unified, t2_unified] = Tensor::unifyDevice(t1, t2);

    Tensor result = Tensor(t1_unified.shape, device_result);

    switch (device_result) {
        case TensorDevice::CPU:
            for (int i = 0; i < t1_unified.size(); i++) {
                result.data->space[i] = t1_unified.data->space[i] - t2_unified.data->space[i];
            }
        break;
        case TensorDevice::GPU:
            TensorKernel::sub_gpu<<<CudaGetBlocks(t1_unified.size()), kCudaThreadsNum>>>(t1_unified.data->space, t2_unified.data->space, result.data->space, t1_unified.size());
        cudaDeviceSynchronize();
        break;
    }

    return result;
}

Tensor operator*(const Tensor &t1, const Tensor &t2) {
    if (t1.shape != t2.shape) {
        throw std::runtime_error("Shapes do not match");
    }

    auto [device_result, t1_unified, t2_unified] = Tensor::unifyDevice(t1, t2);

    Tensor result = Tensor(t1_unified.shape, device_result);

    switch (device_result) {
        case TensorDevice::CPU:
            for (int i = 0; i < t1_unified.size(); i++) {
                result.data->space[i] = t1_unified.data->space[i] * t2_unified.data->space[i];
            }
        break;
        case TensorDevice::GPU:
            TensorKernel::pt_mul_gpu<<<CudaGetBlocks(t1_unified.size()), kCudaThreadsNum>>>(t1_unified.data->space, t2_unified.data->space, result.data->space, t1_unified.size());
        cudaDeviceSynchronize();
        break;
    }

    return result;
}

Tensor operator*(const TensorDataType scalar, const Tensor &tensor) {
    Tensor result = Tensor(tensor.shape, tensor.device);

    switch (tensor.device) {
        case TensorDevice::CPU:
            for (int i = 0; i < tensor.size(); i++) {
                result.data->space[i] = scalar * tensor.data->space[i];
            }
            break;
        case TensorDevice::GPU:
            TensorKernel::scalar_mul_gpu<<<CudaGetBlocks(tensor.size()), kCudaThreadsNum>>>(scalar, tensor.data->space, result.data->space, tensor.size());
            cudaDeviceSynchronize();
            break;
    }

    return result;
}

Tensor operator/(const Tensor &t1, const Tensor &t2) {
    if (t1.shape != t2.shape) {
        throw std::runtime_error("Shapes do not match");
    }

    auto [device_result, t1_unified, t2_unified] = Tensor::unifyDevice(t1, t2);

    Tensor result = Tensor(t1_unified.shape, device_result);

    switch (device_result) {
        case TensorDevice::CPU:
            for (int i = 0; i < t1_unified.size(); i++) {
                result.data->space[i] = t1_unified.data->space[i] / t2_unified.data->space[i];
            }
        break;
        case TensorDevice::GPU:
            TensorKernel::div_gpu<<<CudaGetBlocks(t1_unified.size()), kCudaThreadsNum>>>(t1_unified.data->space, t2_unified.data->space, result.data->space, t1_unified.size());
            cudaDeviceSynchronize();
        break;
    }

    return result;
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

    throw std::runtime_error("Unreachable code");
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

    throw std::runtime_error("Unreachable code");
}

Tensor Tensor::relu_backward(const Tensor &input, const Tensor &grad) {
    if (input.shape != grad.shape) {
        throw std::runtime_error("Shapes do not match");
    }

    auto [device_result, input_unified, grad_unified] = Tensor::unifyDevice(input, grad);

    Tensor result = Tensor(input_unified.shape, device_result);

    switch (device_result) {
        case TensorDevice::CPU:
            for (int i = 0; i < input_unified.size(); i++) {
                result.data->space[i] = input.data->space[i] > 0 ? grad.data->space[i] : 0;
            }
        break;
        case TensorDevice::GPU:
            TensorKernel::relu_backward_gpu<<<CudaGetBlocks(input_unified.size()), kCudaThreadsNum>>>(input_unified.data->space, grad_unified.data->space, result.data->space, input_unified.size());
            cudaDeviceSynchronize();
        break;
    }

    return result;
}

Tensor Tensor::sigmoid_backward(const Tensor &x, const Tensor &grad) {
    Tensor y = x.sigmoid();

    return grad * y * (ones(x.shape, x.device) - y);
}

__global__ void TensorKernel::relu_gpu(const TensorDataType* in, TensorDataType* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

__global__ void TensorKernel::relu_backward_gpu(const TensorDataType* in, const TensorDataType *grad, TensorDataType* out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in[i] > 0 ? grad[i] : 0;
    }
}

__global__ void TensorKernel::sigmoid_gpu(const TensorDataType *in, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = 1.0 / (1.0 + expf(-in[i]));
    }
}

// __global__ void TensorKernel::sigmoid_backward_gpu(const TensorDataType* in, const TensorDataType * grad, TensorDataType* out, int size) {
//     CUDA_KERNEL_LOOP(i, size) {
//         out[i] =
//     }
// }

__global__ void TensorKernel::add_gpu(const TensorDataType *in1, const TensorDataType *in2, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in1[i] + in2[i];
    }
}

__global__ void TensorKernel::sub_gpu(const TensorDataType *in1, const TensorDataType *in2, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in1[i] - in2[i];
    }
}

__global__ void TensorKernel::pt_mul_gpu(const TensorDataType *in1, const TensorDataType *in2, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in1[i] * in2[i];
    }
}

__global__ void TensorKernel::scalar_mul_gpu(const TensorDataType scalar, const TensorDataType *in, TensorDataType *out,
    int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = scalar * in[i];
    }
}

__global__ void TensorKernel::div_gpu(const TensorDataType *in1, const TensorDataType *in2, TensorDataType *out, int size) {
    CUDA_KERNEL_LOOP(i, size) {
        out[i] = in1[i] / in2[i];
    }
}
