//
// Created by timetraveler314 on 1/18/25.
//

#ifndef OPERATORS_H
#define OPERATORS_H

#include <cassert>
#include "../ndarray/nn.cuh"

namespace Operators {
    class TensorOp : public Op {
        // Tensor operator() (std::vector<Tensor> args) const;
    };

    class Reshape final : public TensorOp {
    public:
        std::vector<int> new_shape;

        explicit Reshape(std::vector<int> new_shape): new_shape(std::move(new_shape)) {}

        std::string name() const override {
            return "<Reshape to " + std::to_string(new_shape.size()) + ">";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            auto& x = args[0];
            return x.reshape(new_shape);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            return {out_grad.reshape(args[0]->realize().getShape())};
        }
    };

    class ScalarProduct final : public TensorOp {
    public:
        TensorDataType scalar;

        explicit ScalarProduct(const TensorDataType scalar): scalar(scalar) {}

        std::string name() const override {
            return "<ScalarProduct by " + std::to_string(scalar) + ">";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            auto& x = args[0];
            return scalar * x;
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            return {scalar * out_grad};
        }
    };

    class ReLU final : public TensorOp {
    public:
        std::string name() const override {
            return "ReLU";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            auto& x = args[0];
            return NdArrayNN::forward_relu(x);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            const NdArray x = args[0]->realize();
            return {NdArrayNN::backward_relu(x, out_grad)};
        }
    };

    class Sigmoid final : public TensorOp {
    public:
        std::string name() const override {
            return "Sigmoid";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            auto& x = args[0];
            return NdArrayNN::forward_sigmoid(x);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            const NdArray x = args[0]->realize();
            return {NdArrayNN::backward_sigmoid(x, out_grad)};
        }
    };

    class Linear final : public TensorOp {
    public:
        std::string name() const override {
            return "Linear";
        }

        // Input, weight, bias
        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 3);
            return NdArrayNN::forward_fc(args[0], args[1], args[2]);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 3);
            const NdArray x = args[0]->realize();
            const NdArray w = args[1]->realize();
            const NdArray b = args[2]->realize();

            auto [dx, dw, db] = NdArrayNN::backward_fc(out_grad, x, w);

            return {dx, dw, db};
        }
    };

    class Transpose final : public TensorOp {
    public:
        std::string name() const override {
            return "Transpose";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            return args[0].transpose();
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            return {out_grad.transpose()};
        }
    };

    class Negate final : public TensorOp {
    public:
        std::string name() const override {
            return "Negate";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            return -args[0];
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            return {-out_grad};
        }
    };

    class SoftmaxCrossEntropy final : public TensorOp {
    public:
        std::string name() const override {
            return "SoftmaxCrossEntropy";
        }

        // Args: input, ground truth
        NdArray compute(std::vector<NdArray> &args) const override {
            assert(args.size() == 2);
            return NdArrayNN::cross_entropy(NdArrayNN::forward_softmax(args[0]), args[1]);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 2);
            const NdArray x = args[0]->realize();
            const NdArray y = args[1]->realize();

            auto dx = NdArrayNN::backward_softmax_cross_entropy(NdArrayNN::forward_softmax(x), y);
            return {dx, NdArray::zeros_like(y)};
        }
    };

    /* Non-unary operators */

    class EWiseAdd final : public TensorOp {
    public:
        std::string name() const override {
            return "EWiseAdd";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 2);
            return args[0] + args[1];
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 2);
            return {out_grad, out_grad};
        }
    };

    class MatMul final : public TensorOp {
    public:
        std::string name() const override {
            return "MatMul";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 2);
            return args[0] % args[1];
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 2);
            const NdArray a = args[0]->realize();
            const NdArray b = args[1]->realize();

            auto da = out_grad % b.transpose();
            auto db = a.transpose() % out_grad;

            return {da, db};
        }
    };

    class Conv2D_3x3 final : public TensorOp {
    public:
        std::string name() const override {
            return "Conv2D_3x3";
        }

        // Args: input, kernels
        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 2);
            return NdArrayNN::conv2d(3, 1, 1, args[0], args[1]);
        }

        /* @param out_grad: gradient of the loss wrt the output of the operator
         * @param args: input, kernels
         * @return: gradient of the loss wrt the input and the kernels
         */
        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 2);
            const NdArray x = args[0]->realize();
            const NdArray kernels = args[1]->realize();

            auto [dx, dkernels] = NdArrayNN::conv2d_backward(3, 1, 1, x, kernels, out_grad);

            return {dx, dkernels};
        }
    };

    class Conv2D final : public TensorOp {
    public:
        const int kernel_size, stride, padding;

        Conv2D(const int kernel_size, const int stride, const int padding): kernel_size(kernel_size), stride(stride), padding(padding) {}

        std::string name() const override {
            return "Conv2D_Any";
        }

        // Args: input, kernels
        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 2);
            return NdArrayNN::conv2d(kernel_size, stride, padding, args[0], args[1]);
        }

        /* @param out_grad: gradient of the loss wrt the output of the operator
         * @param args: input, kernels
         * @return: gradient of the loss wrt the input and the kernels
         */
        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 2);
            const NdArray x = args[0]->realize();
            const NdArray kernels = args[1]->realize();

            auto [dx, dkernels] = NdArrayNN::conv2d_backward(kernel_size, stride, padding, x, kernels, out_grad);

            return {dx, dkernels};
        }
    };

    class MaxPool2D final : public TensorOp {
    public:
        std::string name() const override {
            return "MaxPool2D";
        }

        NdArray compute(std::vector<NdArray>& args) const override {
            assert(args.size() == 1);
            return NdArrayNN::forward_max_pooling_2x2(args[0]);
        }

        std::vector<NdArray> gradient(const NdArray out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            const NdArray x = args[0]->realize();
            return {NdArrayNN::backward_max_pooling_2x2(out_grad, x)};
        }
    };
}

#endif //OPERATORS_H
