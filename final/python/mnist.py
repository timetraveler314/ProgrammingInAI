import numpy as np
import torch
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm  # 导入 tqdm

new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

from Designant import *


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

    def accuracy(self, predictions, targets):
        # 计算准确率
        predicted = predictions.numpy().argmax(axis=1)
        correct = (predicted == targets.numpy()).sum()
        return correct / targets.shape()[0]


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

    # 训练开始
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0

        # 使用 tqdm 包装数据加载器，以显示进度条
        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for batch_idx, (data, target) in pbar:
                # 将数据从numpy转换为自定义的Tensor
                data = data.view(min(batch_size, data.shape[0]), -1)
                target = target.numpy()

                data_tensor = Tensor.from_numpy(data, True)
                target_tensor = Tensor.from_numpy(target, False)

                # 前向传播
                my_z1, predictions = net.forward(data_tensor)

                # 计算损失
                loss = net.loss(predictions, target_tensor)

                # 反向传播
                loss.backward()

                # 更新参数
                for layer in [net.fc1, net.fc2]:
                    layer.weight.update(layer.weight - lr * layer.weight.grad())
                    layer.bias.update(layer.bias - lr * layer.bias.grad())

                # 累计损失
                total_loss += loss.numpy()

                # 计算准确率
                accuracy = net.accuracy(predictions, target_tensor)
                correct += accuracy * batch_size
                total_samples += batch_size

                # 更新进度条
                pbar.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=correct / total_samples)

        print(
            f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}, Accuracy: {correct / total_samples}")


if __name__ == "__main__":
    train_mnist()
