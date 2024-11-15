//
// Created by timetraveler314 on 10/28/24.
//

#ifndef DEVICE_SPACE_H
#define DEVICE_SPACE_H
#include <memory>

typedef float TensorDataType;

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

    device_ptr copy_to(const TensorDevice device) const {
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
};

#endif //DEVICE_SPACE_H
