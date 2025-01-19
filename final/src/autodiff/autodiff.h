//
// Created by timetraveler314 on 1/19/25.
//

#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "../tensor.h"

void compute_gradients(const Tensor& out_tensor, const Tensor& out_grad);

// std::vector<Value> find_topological_order(const std::vector<Value>& values);

#endif //AUTODIFF_H
