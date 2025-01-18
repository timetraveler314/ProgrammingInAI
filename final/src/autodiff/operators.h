//
// Created by timetraveler314 on 1/18/25.
//

#ifndef OPERATORS_H
#define OPERATORS_H

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

        std::vector<NdArray> gradient(const Value out_grad, std::vector<Value>& args) const override {
            assert(args.size() == 1);
            const NdArray x = args[0]->realize();
            const NdArray grad = out_grad->realize();
            return {NdArrayNN::backward_relu(x, grad)};
        }
    };
}

#endif //OPERATORS_H
