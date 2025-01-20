//
// Created by timetraveler314 on 1/18/25.
//

#ifndef TENSOR_H
#define TENSOR_H

#include <map>

#include "autodiff/value.h"
#include "autodiff/op.h"
#include "autodiff/operators.h"

// using Tensor = std::shared_ptr<TensorImpl>;
extern int count;
extern std::map<std::string, int> categories;

class TensorImpl : public ValueImpl {
    std::optional<NdArray> grad;

public:
    TensorImpl(const std::vector<int>& shape, Device device, const bool require_grad = false)
        : ValueImpl(NdArray(shape, device), require_grad), grad(NdArray(shape, device)) {
        count++;
        categories["<leaf>"]++;
    }

    // Create a leaf value with the given data
    explicit TensorImpl(const NdArray& data, const bool require_grad = false)
        : ValueImpl(data, require_grad), grad(NdArray(data.getShape(), data.getDevice())) {
        count++;
        categories["<leaf>"]++;
    }

    // Create a Tensor from operation (Op) and arguments (Value's)
    TensorImpl(std::unique_ptr<Op> op, const std::vector<Value>& args, const bool require_grad = false)
        : ValueImpl(std::move(op), args, require_grad), grad(std::nullopt) {
        count++;
        categories[this->getOp()->name()]++;
    }

    ~TensorImpl() override {
        count--;
        if (this->isLeaf()) {
            categories["<leaf>"]--;
        } else {
            categories[this->getOp()->name()]--;
        }
        // if (count) {
        //     std::cout << "TensorImpl count: " << count << std::endl;
        //     for (const auto& [key, value] : categories) {
        //         std::cout << key << ": " << value << std::endl;
        //     }
        // }
    }

    // static Tensor uniform(const std::vector<int>& shape, const Device device, const bool require_grad = false) {
    //     return std::make_shared<TensorImpl>(NdArray::uniform(shape, device), require_grad);
    // }

    friend class Tensor;
};

class Tensor {
    std::shared_ptr<TensorImpl> impl;

public:
    Tensor(std::shared_ptr<TensorImpl> impl): impl(std::move(impl)) {}

    // User Interface
    explicit Tensor(const NdArray& data, const bool requires_grad = false);
    static Tensor xavier(const std::vector<int>& shape, const Device device, const bool require_grad = false) {
        return Tensor(NdArray::xavier(shape, device), require_grad);
    }

    operator std::shared_ptr<TensorImpl>() const {
        return impl;
    }

    std::shared_ptr<TensorImpl> getImpl() const {
        return impl;
    }

    friend std::ostream& operator<<(std::ostream & lhs, const Tensor & rhs);

    Value to_value() const {
        return std::dynamic_pointer_cast<ValueImpl>(impl);
    }

    void setGrad(NdArray grad) const {
        impl->grad = std::move(grad);
    }

    Tensor grad() const {
        if (!impl->grad.has_value()) throw std::runtime_error("No gradient available");
        return Tensor(impl->grad.value(), false);
    }

    void backward();

    // Comparison operators, for std::map
    bool operator<(const Tensor& other) const {
        return impl < other.impl;
    }

    bool isRequiresGrad() const {
        return impl->isRequiresGrad();
    }

    Tensor detach() const {
        return Tensor(impl->realize(), false);
    }

    /* update - update the tensor with new data, leaving the computation graph behind
     * used for updating the weights of the model */
    void update(const Tensor& new_tensor) const {
        if (!impl->isLeaf()) throw std::runtime_error("Cannot update non-leaf tensor");

        impl->cached_data->copy_from(new_tensor.getImpl()->realize());
    }

    /* Tensor operator/operations */
    Tensor operator+(const Tensor& other) const;
    Tensor operator-() const;
    Tensor operator-(const Tensor& other) const;
    friend Tensor operator*(TensorDataType scalar, const Tensor& tensor);
    Tensor operator%(const Tensor& other) const;

    Tensor transpose() const;
    Tensor reshape(const std::vector<int>& new_shape) const;

    std::vector<int> getShape() const;
    Device getDevice() const;
    std::string toString() const;
};

inline Tensor::Tensor(const NdArray &data, const bool requires_grad): impl(std::make_shared<TensorImpl>(data, requires_grad)) {}

inline std::ostream& operator<<(std::ostream & lhs, const Tensor & rhs) {
    return lhs << rhs.getImpl()->realize();
}

#endif //TENSOR_H
