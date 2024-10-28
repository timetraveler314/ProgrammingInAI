//
// Created by timetraveler314 on 10/28/24.
//

#ifndef TENSORNN_CUH
#define TENSORNN_CUH

#include "tensor.h"

namespace TensorNN {
    inline Tensor forward_fc(const Tensor &input, const Tensor &weight, const Tensor &bias) {
        // Shape check
        if (input.getShape().size() != 2 || weight.getShape().size() != 2 || bias.getShape().size() != 1) {
            throw std::runtime_error("Invalid shape for forward_fc");
        }

        const int N = input.getShape()[0];
        const int Cin = input.getShape()[1];
        const int Cout = weight.getShape()[0];

        if (Cin != weight.getShape()[1] || Cout != bias.getShape()[0]) {
            throw std::runtime_error("Invalid shape for forward_fc");
        }

        auto [device, x, w, b] = Tensor::unifyDevice(input, weight, bias);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for forward_fc");
        }

        Tensor y({N, Cout}, TensorDevice::GPU);
        tensor_kernel::forward_fc_kernel_gpu(x.getRawData(), w.getRawData(), b.getRawData(), y.getRawData(), N, Cin, Cout);
        return y;
    }
}

#endif //TENSORNN_CUH
