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
}

#endif //TENSORNN_H
