//
// Created by timetraveler314 on 1/18/25.
//

#ifndef TENSORNN_H
#define TENSORNN_H
#include "tensor.h"
#include "autodiff/operators.h"

namespace TensorFunctional {
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

    inline Tensor SoftmaxCrossEntropy(const Tensor& x, const Tensor& y) {
        std::vector<Value> args = {x.getImpl(), y.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::SoftmaxCrossEntropy>(), args, true);
    }
}

namespace TensorNN {
    class Linear {
    public:
        const int batch_size, in_features, out_features;
        Tensor weight, bias;

        Linear(int batch_size, int in_features, int out_features): batch_size(batch_size), in_features(in_features), out_features(out_features), weight(nullptr), bias(nullptr) {
            weight = Tensor(NdArray::xavier({out_features, in_features}, Device::GPU), true);
            bias = Tensor(NdArray::zeros({out_features}, Device::GPU), true);
        }

        Tensor operator() (const Tensor& x) const {
            std::vector<Value> args = {x.getImpl(), weight.getImpl(), bias.getImpl()};
            return std::make_shared<TensorImpl>(std::make_unique<Operators::Linear>(), args, true);
        }
    };

    class Conv2D {
    public:
        const int C, K;
        Tensor kernels;

        Conv2D(int C, int K): C(C), K(K), kernels(nullptr) {
            kernels = Tensor(NdArray::xavier({K, C, 3, 3}, Device::GPU), true);
        }

        Tensor operator() (const Tensor& x) const {
            std::vector<Value> args = {x.getImpl(), kernels.getImpl()};
            return std::make_shared<TensorImpl>(std::make_unique<Operators::Conv2D>(), args, true);
        }
    };

    class MaxPool2D {
    public:
        MaxPool2D() {}

        Tensor operator() (const Tensor& x) const {
            std::vector<Value> args = {x.getImpl()};
            return std::make_shared<TensorImpl>(std::make_unique<Operators::MaxPool2D>(), args, true);
        }
    };
}

#endif //TENSORNN_H
