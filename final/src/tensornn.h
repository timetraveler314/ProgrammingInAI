//
// Created by timetraveler314 on 1/18/25.
//

#ifndef TENSORNN_H
#define TENSORNN_H
#include "tensor.h"
#include "autodiff/operators.h"

/* TensorFunctional - a namespace for functional-style operations on tensors */
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

    inline Tensor MaxPool2D(const Tensor& x) {
        std::vector<Value> args = {x.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::MaxPool2D>(), args, true);
    }

    inline Tensor SoftmaxCrossEntropy(const Tensor& x, const Tensor& y) {
        std::vector<Value> args = {x.getImpl(), y.getImpl()};
        return std::make_shared<TensorImpl>(std::make_unique<Operators::SoftmaxCrossEntropy>(), args, true);
    }
}

/* TensorNN - a namespace for neural network layers */
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

    class Conv2D_3x3 {
    public:
        const int C, K;
        Tensor kernels;

        Conv2D_3x3(int C, int K): C(C), K(K), kernels(nullptr) {
            kernels = Tensor(NdArray::xavier({K, C, 3, 3}, Device::GPU), true);
        }

        Tensor operator() (const Tensor& x) const {
            std::vector<Value> args = {x.getImpl(), kernels.getImpl()};
            return std::make_shared<TensorImpl>(std::make_unique<Operators::Conv2D_3x3>(), args, true);
        }
    };

    class Conv2D {
    public:
        const int C, K;
        const int kernel_size, stride, padding;
        Tensor kernels;

        Conv2D(int C, int K, int kernel_size, int stride, int padding): C(C), K(K), kernel_size(kernel_size), stride(stride), padding(padding), kernels(nullptr) {
            kernels = Tensor(NdArray::xavier({K, C, kernel_size, kernel_size}, Device::GPU), true);
        }

        Tensor operator() (const Tensor& x) const {
            std::vector<Value> args = {x.getImpl(), kernels.getImpl()};
            return std::make_shared<TensorImpl>(std::make_unique<Operators::Conv2D>(kernel_size, stride, padding), args, true);
        }
    };
}

#endif //TENSORNN_H
