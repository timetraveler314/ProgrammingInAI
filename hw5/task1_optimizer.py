"""
本文件我们尝试实现一个Optimizer类，用于优化一个简单的双层Linear Network
本次作业主要的内容将会在opti_epoch内对于一个epoch的参数进行优化
分为SGD_epoch和Adam_epoch两个函数，分别对应SGD和Adam两种优化器
其余函数为辅助函数，也请一并填写
和大作业的要求一致，我们不对数据处理和读取做任何要求
因此你可以引入任何的库来帮你进行数据处理和读取
理论上我们也不需要依赖lab5的内容，如果你需要的话，你可以将lab5对应代码copy到对应位置
"""
import random

from task0_autodiff import *
from task0_operators import *
import numpy as np
import torch
from torchvision import datasets
import torchvision.transforms.v2 as v2


def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random numbers uniform between low and high"""
    device = cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate random normal with specified mean and std deviation"""
    device = cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """Generate constant Tensor"""
    device = cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-ones Tensor"""
    return constant(
        *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """Generate all-zeros Tensor"""
    return constant(
        *shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad
    )


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """Generate binary random Tensor"""
    device = cpu() if device is None else device
    array = device.rand(*shape) <= p
    return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """Generate one-hot encoding Tensor"""
    device = cpu() if device is None else device
    return Tensor(
        device.one_hot(n, i.numpy(), dtype=dtype),
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return zeros(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return ones(
        *array.shape, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def parse_mnist():
    """
    读取MNIST数据集，并进行简单的处理，如归一化
    你可以可以引入任何的库来帮你进行数据处理和读取
    所以不会规定你的输入的格式
    但需要使得输出包括X_tr, y_tr和X_te, y_te
    """
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0], std=[1])
    ])

    mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    X_tr = np.array([img for img, label in mnist_train]).astype(np.float32)
    # Flatten the images
    X_tr = X_tr.reshape(X_tr.shape[0], -1)
    y_tr = mnist_train.targets.numpy()

    X_te = np.array([img for img, label in mnist_test]).astype(np.float32)
    # Flatten the images
    X_te = X_te.reshape(X_te.shape[0], -1)
    y_te = mnist_test.targets.numpy()

    print(X_tr.shape, y_tr.shape, X_te.shape, y_te.shape)  # (60000, 784) (60000,) (10000, 784) (10000,)

    return X_tr, y_tr, X_te, y_te


def set_structure(n, hidden_dim, k):
    """
    定义你的网络结构，并进行简单的初始化
    一个简单的网络结构为两个Linear层，中间加上ReLU
    Args:
        n: input dimension of the data.
        hidden_dim: hidden dimension of the network.
        k: output dimension of the network, which is the number of classes.
    Returns:
        List of Weights matrix.
    Example:
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    return list(W1, W2)
    """

    # W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    # W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)
    W1 = randn(n, hidden_dim, mean=0, std=1 / np.sqrt(hidden_dim), requires_grad=True)
    W2 = randn(hidden_dim, k, mean=0, std=1 / np.sqrt(k), requires_grad=True)
    return [W1, W2]


def forward(X, weights):
    """
    使用你的网络结构，来计算给定输入X的输出
    Args:
        X : 2D input array of size (num_examples, input_dim).
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
    Returns:
        Z calculated by your network structure.
    Example:
    W1 = weights[0]
    W2 = weights[1]
    return np.maximum(X@W1,0)@W2
    """
    W1 = weights[0]
    W2 = weights[1]
    Z1 = X @ W1
    A1 = relu(Z1)
    Z2 = A1 @ W2
    return Z2


def softmax_loss(Z, y):
    """ 
    一个写了很多遍的Softmax loss...

    Args:
        Z : 2D numpy array of shape (batch_size, num_classes), 
        containing the logit predictions for each class.
        y : 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    # Normalize
    # Z -= np.max(Z, axis=1, keepdims=True)
    #
    # Z_exp = np.exp(Z)
    # Z_sum = np.sum(Z_exp, axis=1, keepdims=True)
    # losses = np.log(Z_sum + 1e-8) - Z[np.arange(Z.shape[0]), y]
    # avg_loss = np.mean(losses)

    y_one_hot = one_hot(Z.shape, i=y)
    x = exp(Z).sum((1,))
    y = log(x).sum()
    z = (Z * y_one_hot).sum()
    loss = (y - z) / Z.shape[0]
    return loss


def opti_epoch(X, y, weights, lr=0.1, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """
    优化一个epoch
    具体请参考SGD_epoch 和 Adam_epoch的代码
    """
    if using_adam:
        Adam_epoch(X, y, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2)
    else:
        SGD_epoch(X, y, weights, lr=lr, batch=batch)


def SGD_epoch(X, y, weights, lr=0.1, batch=100):
    """ 
    SGD优化一个List of Weights
    本函数应该inplace地修改Weights矩阵来进行优化
    用学习率简单更新Weights

    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    iters = (X.shape[0] + batch - 1) // batch
    # Shuffle the batch indices for SGD, Stochastic
    iter_range = list(range(iters))
    random.shuffle(iter_range)

    for it in iter_range:
        it_min, it_max = it * batch, min((it + 1) * batch, X.shape[0])
        X_batch, y_batch = Tensor(X[it_min:it_max, :], requires_grad=False), Tensor(y[it_min:it_max],
                                                                                    requires_grad=False)

        # # Forward pass of linear and ReLU
        # Z1 = X_batch @ W1
        # A1 = np.maximum(Z1, 0)
        # Z2 = A1 @ W2
        #
        # # Softmax
        # Z2 -= np.max(Z2, axis=1, keepdims=True)
        # Z2_exp = np.exp(Z2)
        # Z2_sum = np.sum(Z2_exp, axis=1, keepdims=True)
        # Z2_prob = Z2_exp / Z2_sum # Softmax output
        #
        # # Backward pass of softmax loss
        # dZ2 = Z2_prob.copy()
        # dZ2[np.arange(dZ2.shape[0]), y_batch] -= 1
        # dZ2 /= batch
        #
        # # Compute gradients
        # dW2 = A1.T @ dZ2
        # dA1 = dZ2 @ W2.T
        # dZ1 = dA1 * (Z1 > 0)
        # dW1 = X_batch.T @ dZ1

        Z2 = forward(X_batch, weights)
        loss = softmax_loss(Z2, y_batch)

        # Compute gradients
        loss.backward()

        # SGD update
        for weight in weights:
            weight.cached_data -= lr * weight.grad.realize_cached_data()


def Adam_epoch(X, y, weights, lr=0.0001, batch=100, beta1=0.9, beta2=0.999):
    """ 
    ADAM优化一个
    本函数应该inplace地修改Weights矩阵来进行优化
    使用Adaptive Moment Estimation来进行更新Weights
    具体步骤可以是：
    1. 增加时间步 $t$。
    2. 计算当前梯度 $g$。
    3. 更新一阶矩向量：$m = \beta_1 \cdot m + (1 - \beta_1) \cdot g$。
    4. 更新二阶矩向量：$v = \beta_2 \cdot v + (1 - \beta_2) \cdot g^2$。
    5. 计算偏差校正后的一阶和二阶矩估计：$\hat{m} = m / (1 - \beta_1^t)$ 和 $\hat{v} = v / (1 - \beta_2^t)$。
    6. 更新参数：$\theta = \theta - \eta \cdot \hat{m} / (\sqrt{\hat{v}} + \epsilon)$。
    其中$\eta$表示学习率，$\beta_1$和$\beta_2$是平滑参数，
    $t$表示时间步，$\epsilon$是为了维持数值稳定性而添加的常数，如1e-8。
    
    Args:
        X : 2D input array of size (num_examples, input_dim).
        y : 1D class label array of size (num_examples,)
        weights : list of 2D array of layers weights, of shape [(input_dim, hidden_dim)]
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch
        beta1 (float): smoothing parameter for first order momentum
        beta2 (float): smoothing parameter for second order momentum

    Returns:
        None
    """
    iters = (X.shape[0] + batch - 1) // batch
    W1, W2 = weights[0].detach(), weights[1].detach()

    # Initialize first and second moment estimates
    mW1, mW2 = zeros_like(W1), zeros_like(W2)
    vW1, vW2 = zeros_like(W1), zeros_like(W2)
    t = 0
    epsilon = 1e-8

    for it in range(iters):
        t += 1  # Increase time step

        it_min, it_max = it * batch, min((it + 1) * batch, X.shape[0])
        X_batch, y_batch = Tensor(X[it_min:it_max, :], requires_grad=False), Tensor(y[it_min:it_max],
                                                                                    requires_grad=False)

        # Forward pass of linear and ReLU
        Z1 = forward(X_batch, weights)
        loss = softmax_loss(Z1, y_batch)

        # Compute gradients
        loss.backward()
        dW1 = weights[0].grad.detach()
        dW2 = weights[1].grad.detach()

        # Adam update for W1
        mW1 = beta1 * mW1 + (1 - beta1) * dW1
        vW1 = beta2 * vW1 + (1 - beta2) * (dW1 ** 2)
        mW1_hat = mW1 / (1 - beta1 ** t)
        vW1_hat = vW1 / (1 - beta2 ** t)
        W1 -= lr * mW1_hat / ((vW1_hat ** 0.5) + epsilon)
        weights[0].cached_data = W1.realize_cached_data()

        # Adam update for W2
        mW2 = beta1 * mW2 + (1 - beta1) * dW2
        vW2 = beta2 * vW2 + (1 - beta2) * (dW2 ** 2)
        mW2_hat = mW2 / (1 - beta1 ** t)
        vW2_hat = vW2 / (1 - beta2 ** t)
        W2 -= lr * mW2_hat / ((vW2_hat ** 0.5) + epsilon)
        weights[1].cached_data = W2.realize_cached_data()


def loss_err(h, y):
    """ 
    计算给定预测结果h和真实标签y的loss和error
    """
    return softmax_loss(h, y).realize_cached_data(), np.mean(
        h.realize_cached_data().argmax(axis=1) != y.realize_cached_data())


def train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=500,
             epochs=10, lr=0.5, batch=100, beta1=0.9, beta2=0.999, using_adam=False):
    """ 
    训练过程
    """
    n, k = X_tr.shape[1], y_tr.max() + 1
    weights = set_structure(n, hidden_dim, k)
    np.random.seed(0)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        opti_epoch(X_tr, y_tr, weights, lr=lr, batch=batch, beta1=beta1, beta2=beta2, using_adam=using_adam)
        train_loss, train_err = loss_err(forward(Tensor(X_tr), weights), Tensor(y_tr))
        test_loss, test_err = loss_err(forward(Tensor(X_te), weights), Tensor(y_te))
        print("|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |" \
              .format(epoch, train_loss, train_err, test_loss, test_err))


if __name__ == "__main__":
    X_tr, y_tr, X_te, y_te = parse_mnist()
    weights = set_structure(X_tr.shape[1], 100, y_tr.max() + 1)
    ## using SGD optimizer 
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr=0.2, batch=100, beta1=0.9, beta2=0.999,
             using_adam=False)
    ## using Adam optimizer
    train_nn(X_tr, y_tr, X_te, y_te, weights, hidden_dim=100, epochs=20, lr=0.001, batch=1000, beta1=0.9, beta2=0.999,
             using_adam=True)
