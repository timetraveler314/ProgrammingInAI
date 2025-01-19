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
}

#endif //OPERATORS_H
