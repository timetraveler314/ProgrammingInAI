import code

import numpy as np
import torch

from torchvision import datasets
from torchvision.transforms import v2

new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

from Designant import *

import numpy as np

# 定义网络的参数
n, h, o = 784, 128, 10  # 输入、隐藏层和输出层的大小
np.random.seed(42)

W1 = np.random.randn(h, n) * 0.01
b1 = np.zeros((h, 1))
W2 = np.random.randn(o, h) * 0.01
b2 = np.zeros((o, 1))


# 前向传播的计算
def forward(x):
    z1 = np.dot(x, W1.T) + b1
    a1 = np.maximum(0, z1)  # ReLU激活
    z2 = np.dot(a1, W2.T) + b2
    a2 = softmax(z2)  # Softmax输出
    return a1, a2, z1, z2


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # 稳定性修正
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)


# 损失函数（交叉熵损失）
def compute_loss(y, a2):
    m = y.shape[0]
    loss = -np.sum(np.log(a2) * y) / m  # 平均损失
    return loss


# 反向传播
def backward(x, y, a1, a2, z1, z2):
    m = y.shape[0]  # 样本数量

    # 计算输出层的梯度
    dz2 = a2 - y  # Softmax的梯度
    dW2 = np.dot(dz2.T, a1) / m  # 权重的梯度
    db2 = np.sum(dz2, axis=0, keepdims=True) / m  # 偏置的梯度

    # 计算隐藏层的梯度
    dz1 = np.dot(dz2, W2) * (z1 > 0)  # ReLU的梯度
    dW1 = np.dot(dz1.T, x) / m  # 权重的梯度
    db1 = np.sum(dz1, axis=0, keepdims=True) / m  # 偏置的梯度

    return dW1, db1, dW2, db2

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        # 初始化两层线性网络
        self.fc1 = nnn.Linear(batch_size, input_size, hidden_size)
        self.fc2 = nnn.Linear(batch_size, hidden_size, output_size)

    def forward(self, x):
        # 前向传播
        z1 = self.fc1(x)  # 第一个线性层
        a1 = nnn.functional.relu(z1)  # ReLU 激活
        z2 = self.fc2(a1)  # 第二个线性层
        return z1, z2

    def loss(self, predictions, targets):
        # 计算Softmax交叉熵损失
        return nnn.functional.softmax_cross_entropy(predictions, targets)

def train_mnist():
    # 数据参数
    input_size = 28 * 28  # MNIST图像大小
    hidden_size = 128  # 隐藏层单元数
    output_size = 10  # MNIST类别数
    batch_size = 128  # 每次训练的样本数

    # 加载MNIST数据
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], std=[1])])
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # 初始化网络
    net = SimpleNet(input_size, hidden_size, output_size, batch_size)

    # 优化器参数
    lr = 0.1
    epochs = 10

    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Batch {batch_idx + 1}/{len(train_loader)}")
            # 将数据从numpy转换为自定义的Tensor
            data = data.view(min(batch_size, data.shape[0]), -1)
            target = target.numpy()
            # print(target)

            data_tensor = Tensor.from_numpy(data, True)
            target_tensor = Tensor.from_numpy(target, False)

            # 前向传播
            my_z1, predictions = net.forward(data_tensor)

            # 计算损失
            loss = net.loss(predictions, target_tensor)

            # 反向传播
            loss.backward()

            # 以 NumPy 为 baseline, 检查梯度是否正确
            # global W1, b1, W2, b2
            # W1 = net.fc1.weight.numpy()
            # b1 = net.fc1.bias.numpy()
            # W2 = net.fc2.weight.numpy()
            # b2 = net.fc2.bias.numpy()
            # a1, a2, z1, z2 = forward(data)

            def softmax_np(x):
                e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
                return e_x / np.sum(e_x, axis=1, keepdims=True)

            def cross_entropy_loss_np(softmax_output, labels):
                N = softmax_output.shape[0]
                log_likelihood = -np.log(softmax_output[np.arange(N), labels])
                return np.sum(log_likelihood) / N

            def combined_backward(softmax_output, labels):
                grad = softmax_output.copy()
                grad[np.arange(batch_size), labels] -= 1
                return grad

            # # Predictions Comparison
            # np.testing.assert_allclose(predictions.numpy(), z2, atol=1e-6)
            #
            # y = np.eye(10)[target]
            #
            # loss_numpy = cross_entropy_loss_np(softmax_np(z2), target)
            #
            # dW1, db1, dW2, db2 = backward(data, y, a1, a2, z1, z2)

            # Print max in gradient
            # print("Max in gradient")
            # print(np.max(dW1))
            # print(np.max(net.fc1.weight.grad().numpy()))
            # np.testing.assert_allclose(dW2, net.fc2.weight.grad().numpy(), rtol=1e-4)
            # 更新参数
            for layer in [net.fc1, net.fc2]:
                layer.weight.update(layer.weight - lr * layer.weight.grad())
                layer.bias.update(layer.bias - lr * layer.bias.grad())

            # 累计损失
            total_loss += loss.numpy()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

if __name__ == "__main__":
    train_mnist()