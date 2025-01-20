#include <iostream>

#include "ndarray/ndarray.h"

#include "tensornn.h"

#include "tensor.h"

using namespace std;

int main() {
    // 这是一个非常简单的示例，展示了如何使用我们的张量库进行前向传播、反向传播和梯度下降
    // 具体的使用参考 Python 版本的示例

    // 初始化输入数据（3个样本，4个特征）
    TensorDataType x_data[] = {0.01, 0.02, 0.03, 0.04,
                                0.05, 0.06, 0.07, 0.08,
                                0.09, 0.10, 0.11, 0.12};

    // 权重（4个特征，2个输出单元）
    TensorDataType w_data[] = {1, 1,
                                1, 1,
                                1, 1,
                                1, 1};

    // 偏置（2个输出单元）
    TensorDataType b_data[] = {0, 0, 0, 0, 0, 0};

    Tensor x {NdArray::from_raw_data({3, 4}, Device::GPU, x_data), false};
    Tensor w {NdArray::from_raw_data({4, 2}, Device::GPU, w_data), true};
    Tensor b {NdArray::from_raw_data({3, 2}, Device::GPU, b_data), true};

    Tensor y = x % w + b;

    Tensor relu_y = TensorFunctional::ReLU(y);

    Tensor sigmoid_y = TensorFunctional::Sigmoid(relu_y);

    cout << sigmoid_y << endl;

    // 假设目标值（我们用一个简单的目标张量进行示范）
    TensorDataType target_data[] = {0.5, 0.5, 0.7, 0.7, 0.9, 0.9};  // 目标值
    Tensor target {NdArray::from_raw_data({3, 2}, Device::GPU, target_data), false};

    // 简单的损失，测试自动微分是否正常工作
    Tensor loss = (sigmoid_y - target);

    // 反向传播
    loss.backward();

    cout << "Gradient with respect to weights:" << endl;
    cout << w.grad() << endl;

    cout << "Gradient with respect to biases:" << endl;
    cout << b.grad() << endl;

    return 0;
}