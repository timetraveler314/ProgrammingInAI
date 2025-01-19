//
// Created by timetraveler314 on 1/18/25.
//

#ifndef TENSORNN_H
#define TENSORNN_H
#include "tensor.h"
#include "autodiff/operators.h"

namespace TensorNN {
    inline Tensor ReLU(const Tensor& x) {
        std::vector<Value> args = {x.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::ReLU>(), args, true);
    }

    inline Tensor Sigmoid(const Tensor& x) {
        std::vector<Value> args = {x.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::Sigmoid>(), args, true);
    }

    inline Tensor Linear(const Tensor& x, const Tensor& w, const Tensor& b) {
        std::vector<Value> args = {x.getImpl(), w.getImpl(), b.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::Linear>(), args, true);
    }
}

#endif //TENSORNN_H
