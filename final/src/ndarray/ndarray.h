//
// Created by timetraveler314 on 9/22/24.
//

#ifndef NDARRAY_H
#define NDARRAY_H
#include <vector>
#include <iomanip>

#include <functional>
#include <iostream>

#include "device_space.hpp"

/* NdArray - a class representing a multi-dimensional array
 * that is used for data computation in the neural network
 *
 * this is the original Tensor in homework 3, and is now renamed to NdArray
 *
 * Philosophy:
 * - The NdArray class is designed to be pythonic,
 *   i.e. it is a `reference` to the data, not the data itself
 *   (so `shared_ptr` is used and shallow copy is the default)
 */
class NdArray {
    Device device;
    std::vector<int> shape;
    device_ptr data;

    /* useful template functions to unify the device of multiple tensors */
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

    /* static initializers */
    static NdArray zeros_like(const NdArray & nds);
    static NdArray zeros(std::vector<int> shape, Device device);
    static NdArray ones(std::vector<int> shape, Device device);
    static NdArray iota(std::vector<int> shape, Device device);
    static NdArray uniform(std::vector<int> shape, Device device);
    /* xavier - return a tensor initialized with Xavier initialization */
    static NdArray xavier(const std::vector<int> &shape, Device device);
    static NdArray from_raw_data(std::vector<int> shape, Device device, TensorDataType* data);

    void copy_from(const NdArray &tensor) const;

    NdArray view(const std::vector<int> &newShape) const;
    NdArray reshape(const std::vector<int> &newShape) const;

    TensorDataType* getRawData() const;

    /* cpu - return a copy of the tensor on the CPU */
    NdArray cpu() const;
    /* gpu - return a copy of the tensor on the GPU */
    NdArray gpu() const;

    Device getDevice() const;

    template <typename... Tensors>
    static auto unifyDevice(const Tensors&... tensors) {
        bool hasGpu = ((tensors.device == Device::GPU) || ...);

        if (hasGpu) return std::tuple_cat(std::make_tuple(Device::GPU), unifyToGpu(tensors...));
        else return std::make_tuple(Device::CPU, tensors...);
    }

    /* size - return the number of elements in the tensor */
    int size() const;

    /* operators */
    NdArray operator+(TensorDataType scalar) const;
    friend NdArray operator+(const NdArray& lhs, const NdArray& rhs);
    friend NdArray operator-(const NdArray& lhs, const NdArray& rhs);
    NdArray operator-() const; // Negate
    friend NdArray operator*(const NdArray& lhs, const NdArray& rhs);
    friend NdArray operator*(TensorDataType scalar, const NdArray& tensor);
    friend NdArray operator/(const NdArray& lhs, const NdArray& rhs);
    NdArray operator^(TensorDataType scalar) const; // Power
    NdArray operator/(TensorDataType scalar) const; // Division by scalar

    // Matrix multiplication
    friend NdArray operator%(const NdArray& lhs, const NdArray& rhs);

    NdArray transpose() const;

    void print(std::ostream& os, int depth = 0, int offset = 0) const;
    friend std::ostream& operator<<(std::ostream& os, const NdArray& tensor);

    std::string toString() const;

    std::vector<int> getShape() const;
};


#endif //NDARRAY_H
