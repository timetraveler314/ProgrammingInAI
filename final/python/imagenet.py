from Designant import * #
import torch
from datasets import load_dataset
from torchvision import datasets
from torchvision.transforms import v2
from tqdm import tqdm

class TinyImageNet:
    def __init__(self):
        self.ds = load_dataset("zh-plus/tiny-imagenet")
        self.train_ds = self.ds["train"]
        self.train_ds.set_format("torch", columns=["image", "label"])
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.train_ds)

    def __getitem__(self, idx):
        image = self.train_ds[idx]["image"] / 255.0
        label = self.train_ds[idx]["label"]

        # Handle grayscale images
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)

        image = self.normalize(image)
        return image, label

class TinyImageNetValidation:
    def __init__(self):
        self.ds = load_dataset("zh-plus/tiny-imagenet")
        self.val_ds = self.ds["valid"]
        self.val_ds.set_format("torch", columns=["image", "label"])
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.val_ds)

    def __getitem__(self, idx):
        image = self.val_ds[idx]["image"] / 255.0
        label = self.val_ds[idx]["label"]

        # Handle grayscale images
        if image.shape[0] == 1:
            image = torch.cat([image, image, image], dim=0)

        image = self.normalize(image)
        return image, label

def load_imagenet(batchsize):
    train_dataset = TinyImageNet()
    test_dataset = TinyImageNetValidation()
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batchsize, shuffle=False)

    return train_dataset, train_loader, test_dataset, test_loader

class MyLeNet:
    def __init__(self, batch_size):
        k1 = 64
        k2 = 128
        k3 = 256
        hidden_size = 1024
        self.conv1 = nnn.Conv2d_3x3(3, k1)
        self.conv2 = nnn.Conv2d_3x3(k1, k2)
        # self.conv3 = nnn.Conv2d(k2, k3, 7, 2, 3)
        self.fc1 = nnn.Linear(batch_size, k2 * 16 * 16, hidden_size)
        self.fc2 = nnn.Linear(batch_size, hidden_size, 200)

        self.params = [self.conv1.kernels, self.conv2.kernels, self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias]

    def forward(self, x):
        x = self.conv1(x)
        x = nnn.functional.sigmoid(x)
        x = nnn.functional.maxpool2d(x)
        x = self.conv2(x)
        x = nnn.functional.sigmoid(x)
        x = nnn.functional.maxpool2d(x)
        # shape: [128, 16, 14, 14]
        # x = self.conv3(x)
        # x = nnn.functional.sigmoid(x)
        x = x.reshape([x.shape()[0], x.shape()[1] * x.shape()[2] * x.shape()[3]])
        x = self.fc1(x)
        x = nnn.functional.sigmoid(x)
        x = self.fc2(x)
        return x

    def loss(self, predictions, targets):
        return nnn.functional.softmax_cross_entropy(predictions, targets)

    def accuracy(self, predictions, targets):
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

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = [Tensor.zeros_like(param, False) for param in params]
        self.v = [Tensor.zeros_like(param, False) for param in params]
        self.t = 0

    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            # detach() prevents computation graph from being built
            # if we don't detach, the computation graph will be built
            # and all the intermediate values will be stored in memory
            self.m[i] = (self.beta1 * self.m[i] + (1 - self.beta1) * param.grad()).detach()
            self.v[i] = (self.beta2 * self.v[i] + (1 - self.beta2) * param.grad() ** 2.0).detach()

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            param.update(param - self.lr * m_hat / ((v_hat ** 0.5) + self.eps))


def train_imagenet():
    batch_size = 128  # 每次训练的样本数

    # 优化器参数
    lr = 0.001
    epochs = 100

    # 加载MNIST数据
    train_dataset, train_loader, test_dataset, test_loader = load_imagenet(batch_size)

    net = MyLeNet(batch_size)
    optimizer = SGD(net.params, 0.1)

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

        # 计算测试集准确率
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
    train_imagenet()
