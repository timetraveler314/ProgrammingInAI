//
// Created by timetraveler314 on 10/28/24.
//

#ifndef DEVICE_SPACE_H
#define DEVICE_SPACE_H
#include <memory>

typedef float TensorDataType;

enum class Device {
    CPU,
    GPU
};

struct DeviceSpace {
    Device device;
    TensorDataType* space;
    size_t size;

    DeviceSpace(Device device, size_t size) : device(device), space(nullptr), size(size) {
        switch (device) {
            case Device::CPU:
                space = new TensorDataType[size];
            break;
            case Device::GPU:
                cudaMalloc(&space, size * sizeof(TensorDataType));
            break;
        }
    }

    DeviceSpace(const DeviceSpace& deviceSpace) : device(deviceSpace.device), space(nullptr), size(deviceSpace.size) {
        switch (deviceSpace.device) {
            case Device::CPU:
                space = new TensorDataType[size];
            memcpy(space, deviceSpace.space, size * sizeof(TensorDataType));
            break;
            case Device::GPU:
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
                case Device::CPU:
                    delete [] space;
                break;
                case Device::GPU:
                    cudaFree(space);
                break;
            }
    }
};

class device_ptr : public std::shared_ptr<DeviceSpace> {

public:
    device_ptr(Device device, size_t size) : std::shared_ptr<DeviceSpace>(std::make_shared<DeviceSpace>(device, size)) {}

    device_ptr() = default;
    device_ptr(const device_ptr&) = default;
    device_ptr(device_ptr&&) = default;

    device_ptr & operator=(const device_ptr&) = default;

    device_ptr copy_to(const Device device) const {
        const auto & self = *this;
        auto result = device_ptr(device, self->size);

        switch (device) {
            case Device::CPU:
                switch (self->device) {
                    case Device::CPU:
                        memcpy(result->space, self->space, self->size * sizeof(TensorDataType));
                    break;
                    case Device::GPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToHost);
                    break;
                }
            break;
            case Device::GPU:
                switch (self->device) {
                    case Device::CPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyHostToDevice);
                    break;
                    case Device::GPU:
                        cudaMemcpy(result->space, self->space, self->size * sizeof(TensorDataType), cudaMemcpyDeviceToDevice);
                    break;
                }
            break;
        }

        return result;
    }
};

#endif //DEVICE_SPACE_H
