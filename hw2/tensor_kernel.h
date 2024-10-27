//
// Created by timetraveler314 on 10/25/24.
//

#ifndef TENSOR_KERNEL_H
#define TENSOR_KERNEL_H

#include "tensor.h"

#include <functional>

#include <thrust/transform.h>

namespace tensor_kernel {
    template<typename T>
    concept DeviceIterator = requires (T it)
    {
        { *it } -> std::convertible_to<TensorDataType>;
        { ++it } -> std::same_as<T&>;
    };

    template<typename T>
    concept DeviceUnaryOperation = requires (T op)
    {
        { op(1.0) } -> std::convertible_to<TensorDataType>;
    };

    template<DeviceIterator InputIt, DeviceIterator OutputIt, DeviceUnaryOperation UnaryOp>
    void transform(const TensorDevice device, InputIt first, InputIt last, OutputIt result, UnaryOp op) {
    }
};



#endif //TENSOR_KERNEL_H
