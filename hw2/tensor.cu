//
// Created by timetraveler314 on 9/22/24.
//

#include "tensor.h"
#include "tensor_kernel.h"

device_ptr device_ptr::copy_to(const TensorDevice device) const {
    const auto & self = *this;
    auto result = device_ptr(device, self->size);

    switch (device) {
        case TensorDevice::CPU:
            switch (self->device) {
                case TensorDevice::CPU:
                    memcpy(result->space, self->space, self->size * sizeof(TensorDataType));
                    break;
                case TensorDevice::GPU:
                    cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToHost);
                    break;
            }
            break;
        case TensorDevice::GPU:
            switch (self->device) {
                case TensorDevice::CPU:
                    cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyHostToDevice);
                    break;
                case TensorDevice::GPU:
                    cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
                    break;
            }
            break;
    }

    return result;
}

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
//
// std::tuple<TensorDevice, Tensor, Tensor> Tensor::unifyDevice(const Tensor &lhs, const Tensor &rhs) {
//     if (lhs.device == TensorDevice::GPU || rhs.device == TensorDevice::GPU) {
//         switch (lhs.device) {
//             case TensorDevice::CPU: {
//                 Tensor lhsGpu = lhs.gpu();
//                 return {TensorDevice::GPU, lhsGpu, rhs};
//             }
//             case TensorDevice::GPU: {
//                 switch (rhs.device) {
//                     case TensorDevice::CPU: {
//                         Tensor rhsGpu = rhs.gpu();
//                         return {TensorDevice::GPU, lhs, rhsGpu};
//                     }
//                     case TensorDevice::GPU: {
//                         return {TensorDevice::GPU, lhs, rhs};
//                     }
//                 }
//             }
//         }
//     } else {
//         return {TensorDevice::CPU, lhs, rhs};
//     }
//
//     throw std::runtime_error("Unreachable code");
// }

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

Tensor forward_fc(const Tensor &input, const Tensor &weight, const Tensor &bias) {
    // Shape check
    if (input.getShape().size() != 2 || weight.getShape().size() != 2 || bias.getShape().size() != 1) {
        throw std::runtime_error("Invalid shape for forward_fc");
    }

    const int N = input.getShape()[0];
    const int Cin = input.getShape()[1];
    const int Cout = weight.getShape()[0];

    if (Cin != weight.getShape()[1] || Cout != bias.getShape()[0]) {
        throw std::runtime_error("Invalid shape for forward_fc");
    }

    auto [device, x, w, b] = Tensor::unifyDevice(input, weight, bias);

    if (device != TensorDevice::GPU) {
        throw std::runtime_error("Unimplemented device for forward_fc");
    }

    Tensor y({N, Cout}, TensorDevice::GPU);

    tensor_kernel::forward_fc_kernel_gpu(*x.data, *w.data, *b.data, *y.data, N, Cin, Cout);

    return y;
}
