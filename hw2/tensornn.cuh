//
// Created by timetraveler314 on 10/28/24.
//

#ifndef TENSORNN_CUH
#define TENSORNN_CUH

#include "tensor.h"
#include "tensor_kernel.h"

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

    inline std::tuple<Tensor, Tensor, Tensor> backward_fc(const Tensor &dy, const Tensor &input, const Tensor &weight) {
        // Shape check.
        if (dy.getShape().size() != 2 || input.getShape().size() != 2 || weight.getShape().size() != 2) {
            throw std::runtime_error("Invalid shape for backward_fc");
        }

        const int N = input.getShape()[0];
        const int Cin = input.getShape()[1];
        const int Cout = weight.getShape()[0];

        if (dy.getShape()[0] != N || dy.getShape()[1] != Cout || Cin != weight.getShape()[1]) {
            throw std::runtime_error("Invalid shape for backward_fc");
        }

        auto [device, x, w, b] = Tensor::unifyDevice(input, weight, Tensor({1}, TensorDevice::CPU));

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for backward_fc");
        }

        Tensor dx({N, Cin}, TensorDevice::GPU), dw({Cout, Cin}, TensorDevice::GPU), db({Cout}, TensorDevice::GPU);
        tensor_kernel::backward_fc_kernel_gpu(x.getRawData(), w.getRawData(), dy.getRawData(), dx.getRawData(), dw.getRawData(), db.getRawData(), N, Cin, Cout);
        return {dx, dw, db};
    }

    inline Tensor forward_softmax(const Tensor &input) {
        if (input.getShape().size() != 2) {
            throw std::runtime_error("Invalid shape for forward_softmax");
        }

        const int N = input.getShape()[0];
        const int C = input.getShape()[1];

        auto [device, x] = Tensor::unifyDevice(input);

        if (device != TensorDevice::GPU) {
            throw std::runtime_error("Unimplemented device for forward_softmax");
        }

        Tensor y({N, C}, TensorDevice::GPU);
        tensor_kernel::forward_softmax_kernel_gpu(x.getRawData(), y.getRawData(), N, C);

        return y;
    }
}

#endif //TENSORNN_CUH
