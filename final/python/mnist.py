from Designant import * #
import torch
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

new_mirror = 'https://ossci-datasets.s3.amazonaws.com/mnist'
datasets.MNIST.resources = [
    ('/'.join([new_mirror, url.split('/')[-1]]), md5)
    for url, md5 in datasets.MNIST.resources
]

class SimpleNet:
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        # 初始化两层线性网络
        self.fc1 = nnn.Linear(batch_size, input_size, hidden_size)
        self.fc2 = nnn.Linear(batch_size, hidden_size, output_size)

        self.params = [self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def forward(self, x):
        # 前向传播
        x = x.reshape([x.shape()[0], x.shape()[1] * x.shape()[2] * x.shape()[3]])
        z1 = self.fc1(x)  # 第一个线性层
        a1 = nnn.functional.relu(z1)  # ReLU 激活
        z2 = self.fc2(a1)  # 第二个线性层
        return z2

    def loss(self, predictions, targets):
        # 计算Softmax交叉熵损失
        return nnn.functional.softmax_cross_entropy(predictions, targets)

    def accuracy(self, predictions, targets):
        # 计算准确率
        predicted = predictions.numpy().argmax(axis=1)
        correct = (predicted == targets.numpy()).sum()
        return correct / targets.shape()[0]


class ConvNet:
    # Structure: Conv(k1) -> ReLU -> MaxPool -> Conv(k2) -> ReLU -> MaxPool -> Reshape -> Linear -> ReLU -> Linear
    def __init__(self, N):
        k1 = 64
        k2 = 64
        self.conv1 = nnn.Conv2d(1, k1)
        self.conv2 = nnn.Conv2d(k1, k2)
        self.fc1 = nnn.Linear(N, 7 * 7 * k2, 128)
        self.fc2 = nnn.Linear(N, 128, 10)

        self.params = [self.conv1.kernels, self.conv2.kernels, self.fc1.weight,
                       self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def forward(self, x):
        x = self.conv1(x)
        x = nnn.functional.relu(x)
        x = nnn.functional.maxpool2d(x)
        x = self.conv2(x)
        x = nnn.functional.relu(x)
        x = nnn.functional.maxpool2d(x)
        x = x.reshape([x.shape()[0], x.shape()[1] * x.shape()[2] * x.shape()[3]])
        x = self.fc1(x)
        x = nnn.functional.relu(x)
        x = self.fc2(x)
        return x

    def loss(self, predictions, targets):
        # 计算Softmax交叉熵损失
        return nnn.functional.softmax_cross_entropy(predictions, targets)

    def accuracy(self, predictions, targets):
        # 计算准确率
        predicted = predictions.numpy().argmax(axis=1)
        correct = (predicted == targets.numpy()).sum()
        return correct / targets.shape()[0]


class SGD:
    def __init__(self, params, lr):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.update(param - self.lr * param.grad())


def train_mnist():
    # 数据参数
    input_size = 28 * 28  # MNIST图像大小
    hidden_size = 128  # 隐藏层单元数
    output_size = 10  # MNIST类别数
    batch_size = 128  # 每次训练的样本数

    # 优化器参数
    lr = 0.1
    epochs = 10

    # 加载MNIST数据
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], std=[1])])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    print("MNIST Loaded, with {} training samples and {} testing samples".format(len(train_dataset), len(test_dataset)))

    # Network
    #net = SimpleNet(input_size, hidden_size, output_size, batch_size)
    net = ConvNet(batch_size)
    optimizer = SGD(net.params, lr)

    # 训练
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total_samples = 0

        with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
            for batch_idx, (data, target) in pbar:
                # 将数据读入我的 Tensor 框架
                data_tensor = Tensor.from_numpy(data, True)
                target_tensor = Tensor.from_numpy(target, False)

                # Forward
                logits = net.forward(data_tensor)

                # Loss and Backward
                loss = net.loss(logits, target_tensor)
                loss.backward()

                # Update
                optimizer.step()

                # Accuracy
                total_loss += loss.numpy()
                accuracy = net.accuracy(logits, target_tensor)
                correct += accuracy * batch_size
                total_samples += batch_size

                pbar.set_postfix(loss=total_loss / (batch_idx + 1), accuracy=correct / total_samples)

        test_accuracy = 0
        for test_data, test_target in test_loader:
            test_data_tensor = Tensor.from_numpy(test_data, False)
            test_target_tensor = Tensor.from_numpy(test_target, False)
            test_logits = net.forward(test_data_tensor)
            test_loss = net.loss(test_logits, test_target_tensor)
            test_accuracy += net.accuracy(test_logits, test_target_tensor) * test_data.shape[0]

        test_accuracy /= len(test_dataset)

        print(f"Finished Epoch {epoch + 1}/{epochs}, Train Loss: {total_loss / len(train_loader)}, Test Accuracy: {test_accuracy}")


if __name__ == "__main__":
    train_mnist()
