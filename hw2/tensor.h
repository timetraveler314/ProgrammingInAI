//
// Created by timetraveler314 on 9/22/24.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <iostream>
#include <iomanip>
#include <memory>

#include <cuda_runtime.h>
#include <functional>

#include <thrust/transform.h>
#include <thrust/execution_policy.h>

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

    DeviceSpace(const DeviceSpace& deviceSpace) : device(deviceSpace.device), space(nullptr), size(deviceSpace.size) {
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

    operator TensorDataType*() const {
        return space;
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

    device_ptr copy_to(const TensorDevice device) const;
};

class Tensor {
    TensorDevice device;
    std::vector<int> shape;
    device_ptr data;

public:
    Tensor(std::vector<int> shape, TensorDevice device);
    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) = default;

    static Tensor ones(std::vector<int> shape, TensorDevice device);

    // Nothing needed here. Data pointer will be freed automatically by
    // the shared_ptr managing DeviceSpace.
    ~Tensor() = default;

    Tensor cpu() const;
    Tensor gpu() const;
    static std::tuple<TensorDevice, Tensor, Tensor> unifyDevice(const Tensor& lhs, const Tensor& rhs);

    int size() const;

    template<typename UnaryOperation>
    Tensor transform(UnaryOperation op) const {
        Tensor result(shape, device);
        switch (device) {
            case TensorDevice::CPU:
                thrust::transform(thrust::host, data->space, data->space + size(), result.data->space, op);
            case TensorDevice::GPU:
                thrust::transform(thrust::device, data->space, data->space + size(), result.data->space, op);
        }

        return result;
    }

    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(TensorDataType scalar, const Tensor& tensor);
    friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

    void print(std::ostream& os, int depth = 0, int offset = 0) const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    std::vector<int> getShape() const;
};


#endif //TENSOR_H
