//
// Created by timetraveler314 on 1/18/25.
//

#include "tensor.h"

#include <autodiff.h>

void Tensor::backward() {
    this->to_value()->realize();

    if (!this->impl->grad) {
        this->impl->grad = NdArray::ones(this->to_value()->cached_data->getShape(), this->to_value()->cached_data->getDevice());
    }
    compute_gradients(*this, Tensor{*this->impl->grad});
}

Tensor Tensor::operator+(const Tensor &other) const {
    std::vector args = {this->to_value(), other.to_value()};

    return std::make_shared<TensorImpl>(std::make_unique<Operators::EWiseAdd>(), args, this->isRequiresGrad() || other.isRequiresGrad());
}

Tensor Tensor::operator-() const {
    return std::make_shared<TensorImpl>(std::make_unique<Operators::Negate>(), std::vector{this->to_value()}, this->isRequiresGrad());
}

// Use + and negate to implement subtraction
Tensor Tensor::operator-(const Tensor &other) const {
    return *this + (-other);
}

Tensor operator*(TensorDataType scalar, const Tensor &tensor) {
    return std::make_shared<TensorImpl>(std::make_unique<Operators::ScalarProduct>(), std::vector{tensor.to_value()}, tensor.isRequiresGrad());
}

Tensor Tensor::operator%(const Tensor &other) const {
    return std::make_shared<TensorImpl>(std::make_unique<Operators::MatMul>(), std::vector{this->to_value(), other.to_value()}, this->isRequiresGrad() || other.isRequiresGrad());
}

Tensor Tensor::transpose() const {
    return std::make_shared<TensorImpl>(std::make_unique<Operators::Transpose>(), std::vector{this->to_value()}, this->isRequiresGrad());
}

Tensor Tensor::reshape(const std::vector<int> &new_shape) const {
    return std::make_shared<TensorImpl>(std::make_unique<Operators::Reshape>(new_shape), std::vector{this->to_value()}, this->isRequiresGrad());
}

std::vector<int> Tensor::getShape() const {
    return impl->realize().getShape();
}

Device Tensor::getDevice() const {
    return impl->realize().getDevice();
}

std::string Tensor::toString() const {
    return impl->realize().toString();
}
