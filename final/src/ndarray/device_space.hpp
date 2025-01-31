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

extern int device_space_count;

/* DeviceSpace - a wrapper around a pointer to a memory space on a device
 * that automatically deallocates the memory when the object goes out of scope
 */
struct DeviceSpace {
    Device device;
    TensorDataType* space;
    size_t size;

    DeviceSpace(const DeviceSpace&) = delete;
    DeviceSpace& operator=(const DeviceSpace&) = delete;

    DeviceSpace(DeviceSpace&& deviceSpace) noexcept : device(deviceSpace.device), space(deviceSpace.space), size(deviceSpace.size) {
        deviceSpace.space = nullptr;
    }

    DeviceSpace& operator=(DeviceSpace&& deviceSpace) noexcept {
        if (this != &deviceSpace) {
            device = deviceSpace.device;
            space = deviceSpace.space;
            size = deviceSpace.size;
            deviceSpace.space = nullptr;
        }
        return *this;
    }

    DeviceSpace(Device device, size_t size) : device(device), space(nullptr), size(size) {
        switch (device) {
            case Device::CPU:
                space = new TensorDataType[size];
            break;
            case Device::GPU:
                cudaMalloc(&space, size * sizeof(TensorDataType));
            break;
        }
        device_space_count += size;
    }

    operator TensorDataType*() const {
        return space;
    }

    ~DeviceSpace() {
        // std::cout << "DeviceSpace destructor, size: " << device_space_count << std::endl;
        device_space_count-= size;
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

/* device_ptr - a shared pointer to a DeviceSpace object, i.e. an RC model
 * that provides a copy_to method to copy the data to a different device
 *
 * this is introduced to simplify the grammar of the NdArray class
 */
class device_ptr : public std::shared_ptr<DeviceSpace> {

public:
    device_ptr(Device device, size_t size) : std::shared_ptr<DeviceSpace>(std::make_shared<DeviceSpace>(device, size)) {}

    device_ptr() = default;
    device_ptr(const device_ptr&) = default;
    device_ptr(device_ptr&&) = default;

    device_ptr & operator=(const device_ptr&) = default;
    device_ptr & operator=(device_ptr&&) = default;

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
