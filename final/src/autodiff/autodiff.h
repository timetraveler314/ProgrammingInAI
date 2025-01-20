//
// Created by timetraveler314 on 1/19/25.
//

#ifndef AUTODIFF_H
#define AUTODIFF_H

#include "../tensor.h"

/* compute_gradients - compute the gradients of the loss wrt the input tensors
 * and update the grad field of the tensors in the computational graph
 * (with `requires_grad = true`) to new gradients
 *
 * @param out_tensor: the output tensor of the computation graph
 * @param out_grad: the gradient of the loss wrt the output tensor
 */
void compute_gradients(const Tensor& out_tensor, const Tensor& out_grad);

#endif //AUTODIFF_H
