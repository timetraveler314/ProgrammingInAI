//
// Created by timetraveler314 on 1/18/25.
//

#ifndef TENSOR_H
#define TENSOR_H

#include "autodiff/value.h"
#include "autodiff/op.h"
#include "autodiff/operators.h"

class TensorImpl;

class Tensor {
    std::shared_ptr<TensorImpl> impl;

public:
    Tensor(std::shared_ptr<TensorImpl> impl): impl(std::move(impl)) {}

    // User Interface
    explicit Tensor(const NdArray& data, const bool requires_grad = false);

    operator std::shared_ptr<TensorImpl>() const {
        return impl;
    }

    std::shared_ptr<TensorImpl> getImpl() const {
        return impl;
    }

    friend std::ostream& operator<<(std::ostream & lhs, const Tensor & rhs);
};

// using Tensor = std::shared_ptr<TensorImpl>;

class TensorImpl : public ValueImpl {
    Tensor grad{nullptr};

public:
    TensorImpl(const std::vector<int>& shape, Device device, const bool require_grad = false)
        : ValueImpl(NdArray(shape, device), require_grad) {
        if (require_grad) {
            grad = std::make_shared<TensorImpl>(shape, device, false);
        }
    }

    // Create a leaf value with the given data
    explicit TensorImpl(const NdArray& data, const bool require_grad = false)
        : ValueImpl(data, require_grad) {
        if (require_grad) {
            grad = std::make_shared<TensorImpl>(data.getShape(), data.getDevice(), false);
        }
    }

    // Create a Tensor from operation (Op) and arguments (Value's)
    TensorImpl(std::unique_ptr<Op> op, const std::vector<Value>& args, const bool require_grad = false)
        : ValueImpl(std::move(op), args, require_grad) {
        if (require_grad) {
            grad = std::make_shared<TensorImpl>(args[0]->realize().getShape(), args[0]->realize().getDevice(), false);
        }
    }

    static Tensor uniform(const std::vector<int>& shape, const Device device, const bool require_grad = false) {
        return std::make_shared<TensorImpl>(NdArray::uniform(shape, device), require_grad);
    }
};

inline Tensor::Tensor(const NdArray &data, const bool requires_grad): impl(std::make_shared<TensorImpl>(data, requires_grad)) {}

inline std::ostream& operator<<(std::ostream & lhs, const Tensor & rhs) {
    return lhs << rhs.getImpl()->realize();
}

#endif //TENSOR_H
