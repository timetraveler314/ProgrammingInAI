//
// Created by timetraveler314 on 9/22/24.
//

#ifndef TENSOR_H
#define TENSOR_H
#include <vector>
#include <iomanip>

#include <functional>

#include "device_space.h"

class Tensor {
    TensorDevice device;
    std::vector<int> shape;
    device_ptr data;

    template <typename First, typename... Rest>
    static auto unifyToGpu(const First& first, const Rest&... rest) {
        auto firstGpu = first.device == TensorDevice::GPU ? first : first.gpu();
        auto restResult = unifyToGpu(rest...);
        return std::tuple_cat(std::make_tuple(firstGpu), restResult);
    }

    static auto unifyToGpu(const Tensor& tensor) {
        return std::make_tuple(tensor.device == TensorDevice::GPU ? tensor : tensor.gpu());
    }

public:
    Tensor(std::vector<int> shape, TensorDevice device);
    Tensor(const Tensor& tensor);
    Tensor(Tensor&& tensor) = default;

    static Tensor ones(std::vector<int> shape, TensorDevice device);
    static Tensor iota(std::vector<int> shape, TensorDevice device);
    static Tensor uniform(std::vector<int> shape, TensorDevice device, TensorDataType low = 0.0f, TensorDataType high = 1.0f);

    // Nothing needed here. Data pointer will be freed automatically by
    // the shared_ptr managing DeviceSpace.
    ~Tensor() = default;

    TensorDataType* getRawData() const;

    Tensor cpu() const;
    Tensor gpu() const;

    TensorDevice getDevice() const;

    template <typename... Tensors>
    static auto unifyDevice(const Tensors&... tensors) {
        bool hasGpu = ((tensors.device == TensorDevice::GPU) || ...);

        if (hasGpu) return std::tuple_cat(std::make_tuple(TensorDevice::GPU), unifyToGpu(tensors...));
        else return std::make_tuple(TensorDevice::CPU, tensors...);
    }

    int size() const;

    friend Tensor operator+(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator-(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(const Tensor& lhs, const Tensor& rhs);
    friend Tensor operator*(TensorDataType scalar, const Tensor& tensor);
    friend Tensor operator/(const Tensor& lhs, const Tensor& rhs);

    void print(std::ostream& os, int depth = 0, int offset = 0) const;
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor);

    std::string toString() const;

    std::vector<int> getShape() const;
};


#endif //TENSOR_H
