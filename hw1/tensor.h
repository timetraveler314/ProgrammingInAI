//
// Created by timetraveler314 on 9/22/24.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <iostream>
#include <functional>
#include <iomanip>
#include <memory>

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

struct DeviceSpace {
    TensorDevice device;
    TensorDataType* space;
    size_t size;

    DeviceSpace(TensorDevice device, size_t size) : device(device), space(nullptr), size(size) {
        switch (device) {
            case TensorDevice::CPU:
                space = new TensorDataType[size];
            break;
            case TensorDevice::GPU:
                cudaMalloc(&space, size * sizeof(TensorDataType));
            break;
        }
    }

    DeviceSpace(const DeviceSpace& deviceSpace) : device(device), space(nullptr), size(size) {
        switch (deviceSpace.device) {
            case TensorDevice::CPU:
                space = new TensorDataType[size];
                memcpy(space, deviceSpace.space, size * sizeof(TensorDataType));
            break;
            case TensorDevice::GPU:
                cudaMalloc(&space, size * sizeof(TensorDataType));
                cudaMemcpy(space, deviceSpace.space, size * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
            break;
        }
    }

    ~DeviceSpace() {
        if (space)
            switch (device) {
                case TensorDevice::CPU:
                    delete [] space;
                break;
                case TensorDevice::GPU:
                    cudaFree(space);
                break;
            }
    }
};

class device_ptr : public std::shared_ptr<DeviceSpace> {

public:
    device_ptr(TensorDevice device, size_t size) : std::shared_ptr<DeviceSpace>(std::make_shared<DeviceSpace>(device, size)) {}

    device_ptr() = default;
    device_ptr(const device_ptr&) = default;
    device_ptr(device_ptr&&) = default;

    device_ptr & operator=(const device_ptr&) = default;

    device_ptr copy_to(const TensorDevice device) const {
        const auto & self = *this;
        auto result = device_ptr(device, self->size);

        switch (device) {
            case TensorDevice::CPU:
                switch (self->device) {
                    case TensorDevice::CPU:
                        memcpy(result->space, self->space, self->size * sizeof(TensorDataType));
                    case TensorDevice::GPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToHost);
                }
            break;
            case TensorDevice::GPU:
                switch (self->device) {
                    case TensorDevice::CPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyHostToDevice);
                    case TensorDevice::GPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
                }
            break;
        }

        return result;
    }
};

class Tensor {
public:
    TensorDevice device;
    std::vector<int> shape;
    device_ptr data;

public:
    Tensor(std::vector<int> shape, TensorDevice device);
    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) = default;

    // Nothing needed here. Data pointer will be freed automatically by
    // the shared_ptr managing DeviceSpace.
    ~Tensor() = default;

    Tensor cpu() const;
    Tensor gpu() const;
    static std::tuple<TensorDevice, Tensor, Tensor> unifyDevice(const Tensor& lhs, const Tensor& rhs);

    int size() const;

    Tensor relu() const;
    Tensor sigmoid() const;
    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const TensorDataType scalar, const Tensor& tensor);
    friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

    void acceptModifier(const std::function<void(DeviceSpace&)> &modifier) const;

    void print(std::ostream& os, int depth = 0, int offset = 0) const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);
};

namespace TensorKernel {
    __global__ void relu_gpu(const TensorDataType* in, TensorDataType* out, int size);
    __global__ void sigmoid_gpu(const TensorDataType* in, TensorDataType* out, int size);
    __global__ void add_gpu(const TensorDataType* in1, const TensorDataType* in2, TensorDataType* out, int size);
    __global__ void sub_gpu(const TensorDataType* in1, const TensorDataType* in2, TensorDataType* out, int size);
    __global__ void pt_mul_gpu(const TensorDataType* in1, const TensorDataType* in2, TensorDataType* out, int size);
    __global__ void scalar_mul_gpu(const TensorDataType scalar, const TensorDataType* in, TensorDataType* out, int size);
    __global__ void div_gpu(const TensorDataType* in1, const TensorDataType* in2, TensorDataType* out, int size);
}



#endif //TENSOR_H
