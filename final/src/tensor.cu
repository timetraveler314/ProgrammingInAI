//
// Created by timetraveler314 on 1/18/25.
//

#include "tensor.h"

#include <autodiff.h>

void Tensor::backward(const Tensor &out_grad) {
    compute_gradients(*this, out_grad);
}
