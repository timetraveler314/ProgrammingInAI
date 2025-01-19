//
// Created by timetraveler314 on 9/22/24.
//

#ifndef NDARRAY_H
#define NDARRAY_H
#include <vector>
#include <iomanip>

#include <functional>

#include "device_space.hpp"

class NdArray {
    Device device;
    std::vector<int> shape;
    device_ptr data;

    template <typename First, typename... Rest>
    static auto unifyToGpu(const First& first, const Rest&... rest) {
        auto firstGpu = first.device == Device::GPU ? first : first.gpu();
        auto restResult = unifyToGpu(rest...);
        return std::tuple_cat(std::make_tuple(firstGpu), restResult);
    }

    static auto unifyToGpu(const NdArray& tensor) {
        return std::make_tuple(tensor.device == Device::GPU ? tensor : tensor.gpu());
    }

public:
    NdArray(std::vector<int> shape, Device device);
    NdArray(const NdArray& tensor);
    NdArray(NdArray&& tensor) = default;
    NdArray& operator=(const NdArray& tensor) = default;
    NdArray& operator=(NdArray&& tensor) = default;

    static NdArray zeros_like(const NdArray & nds);
    static NdArray zeros(std::vector<int> shape, Device device);
    static NdArray ones(std::vector<int> shape, Device device);
    static NdArray iota(std::vector<int> shape, Device device);
    static NdArray uniform(std::vector<int> shape, Device device);
    static NdArray xavier(const std::vector<int> &shape, Device device);
    static NdArray from_raw_data(std::vector<int> shape, Device device, TensorDataType* data);

    NdArray view(const std::vector<int> &newShape) const;
    NdArray reshape(const std::vector<int> &newShape) const;

    // Nothing needed here. Data pointer will be freed automatically by
    // the shared_ptr managing DeviceSpace.
    ~NdArray() = default;

    TensorDataType* getRawData() const;

    NdArray cpu() const;
    NdArray gpu() const;

    Device getDevice() const;

    template <typename... Tensors>
    static auto unifyDevice(const Tensors&... tensors) {
        bool hasGpu = ((tensors.device == Device::GPU) || ...);

        if (hasGpu) return std::tuple_cat(std::make_tuple(Device::GPU), unifyToGpu(tensors...));
        else return std::make_tuple(Device::CPU, tensors...);
    }

    int size() const;

    friend NdArray operator+(const NdArray& lhs, const NdArray& rhs);
    friend NdArray operator-(const NdArray& lhs, const NdArray& rhs);
    NdArray operator-() const; // Negate
    friend NdArray operator*(const NdArray& lhs, const NdArray& rhs);
    friend NdArray operator*(TensorDataType scalar, const NdArray& tensor);
    friend NdArray operator/(const NdArray& lhs, const NdArray& rhs);

    // Matrix multiplication
    friend NdArray operator%(const NdArray& lhs, const NdArray& rhs);

    NdArray transpose() const;

    void print(std::ostream& os, int depth = 0, int offset = 0) const;
    friend std::ostream& operator<<(std::ostream& os, const NdArray& tensor);

    std::string toString() const;

    std::vector<int> getShape() const;
};


#endif //NDARRAY_H
