# Programming In Artifical Intelligence - Labs

Labs for the course _[Programming In Artifical Intelligence](https://pkuprogramminginai.github.io/Labs-Documentation/#/)_
in 2024 Fall at Peking University. The project aims to implement a prototype of a neural network framework similar to PyTorch,
including basic tensor operations, simple NN layers, autograd and optimizers. The project is written in CUDA C++ and 
exported to Python using PyBind11.

It's worth noting that...
- the project is not intended to be a full-featured and effective deep learning framework, but a demonstration of the basic ideas.
- my implementation of Tensor is entirely in C++, i.e. autograd is not implemented in Python.

## Labs

- Lab 1: Basic usage of PyTorch
- Lab 2: CUDA kernels for NN operators (including convolution, pooling, etc.) and the wrapper class `Tensor`
- Lab 3: PyBind11 for exporting C++ `Tensor` class to Python and unit tests
- Lab 4: Python class for tensors with autograd
- Lab 5: SGD and Adam optimizers
- Final project: Implement a simple neural network framework for MNIST classification (bonus for (tiny-)imagenet)

## Environment

The labs are tested on Arch Linux with the following environment:
- C++20 standard, CMake 3.31.3
- `libstdc++` installed with `conda install -c conda-forge libstdcxx-ng` for compatibility with C++20
- NVIDIA GeForce RTX 4060 Laptop GPU (8GB) with CUDA 12.7

## Design

Since the final project is a synthesis of all the ideas in the labs, we will introduce the design of the final project here.

### `NdArray` class at `src/ndarray`

The library provides a `NdArray` class, which is an abstraction layer for multi-dimensional arrays and their operations.
`NdArray` can be either on CPU or GPU, implementing tensor operations with CUDA kernels. The class provides a set of NN-related operators so that it can be used as the foundation of
the tensor with autograd.

As for memory management, `NdArray` uses a reference counting mechanism to manage memory. To be specific, the `device_ptr` class (derived from `std::shared_ptr`) 
is used to manage the memory of the tensor.

#### CUDA Kernels

Usual arithmetic operations are omitted here. Instead, we will introduce the kernels for convolution and pooling.
Basically, the project uses `cuBLAS` for GEMMs. However, as a demonstration, no `cuDNN` is used for convolution and pooling, 
instead we implement the kernels ourselves in a rather naive way.

Convolution is implemented through `im2col` and `col2im` operations. Then the convolution kernel becomes a simple matrix multiplication operation. The naive kernels
are written in a "one-thread-one-element" way, with each thread responsible for one element in the original image, slow but easy to understand.

### `Op` and `Value` classes at `src/autograd`

They are interfaces for operations and computational nodes in the computation graph. 

`Op` is the base class for all operations, where each operation has a forward (`compute`) and a backward (`gradient`) method.

`Value = std::shared_ptr<ValueImpl>` represents a computational node in the computation graph,
either a leaf or an intermediate node with references to input nodes. The `ValueImpl` trick
is used to support polymorphism in the `Value` class, since our `Tensor` as an auto-differentiable object is inherited from `ValueImpl`.

### `Tensor` class at `src/tensor.*`

`Tensor <: std::shared_ptr<TensorImpl>` is the class for tensors with autograd. It (actually, `TensorImpl`) is a subclass of `ValueImpl`, holding a gradient `Tensor` to support backpropagation.

To support pure computations without autograd, the `Tensor` class provides a `detach` method to detach the tensor from the computation graph.
This is especially useful in Adam optimizer, where the intermediate computation nodes will explode the memory usage with time.

### NN Layers at `src/tensornn.h`

The project provides a set of NN layers, including `Linear`, `Conv2d`, `MaxPool2d`, `ReLU`, `Sigmoid`, `Softmax`, etc.

## Usage in Python (with optimizers)

The basic interface in Python is designed to be similar to PyTorch. An example of training a simple CNN on MNIST is shown in
the final project, where the `Adam` optimizer is used (`final/python/mnist.py`).

```python
class ConvNet:
    # Structure: Conv(k1) -> ReLU -> MaxPool -> Conv(k2) -> ReLU -> MaxPool -> Reshape -> Linear -> ReLU -> Linear
    def __init__(self, N):
        k1 = 64
        k2 = 64
        self.conv1 = nnn.Conv2d_3x3(1, k1)
        self.conv2 = nnn.Conv2d_3x3(k1, k2)
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
        # Softmax Cross Entropy Loss
        return nnn.functional.softmax_cross_entropy(predictions, targets)

    def accuracy(self, predictions, targets):
        predicted = predictions.numpy().argmax(axis=1)
        correct = (predicted == targets.numpy()).sum()
        return correct / targets.shape()[0]

...
net = ConvNet(batch_size)
optimizer = Adam(net.params, 0.001)

for epoch in range(epochs):
    total_loss = 0
    correct = 0
    total_samples = 0

    with tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs}") as pbar:
        for batch_idx, (data, target) in pbar:
            # Reading data to GPU Tensor
            data_tensor = Tensor.from_numpy(data, True)
            target_tensor = Tensor.from_numpy(target, False)

            # Forward
            logits = net.forward(data_tensor)

            # Loss and Backward
            loss = net.loss(logits, target_tensor)
            loss.backward()

            # Update
            optimizer.step()
...
```