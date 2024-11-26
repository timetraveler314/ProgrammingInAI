from torch.ao.quantization.fx import convert

# 人工智能中的编程 第三次作业实验报告

本作业在 Arch Linux 系统上使用 CUDA 12.7 和 CMake 3.31.0 进行开发，基于 `C++ 20` 标准. Python 版本为 3.12.5.

其中 PyBind 指定编译通过 CMake 实现，具体可见 `src/CMakeLists.txt` 文件.

## 项目结构

```
├── mnist.py # MNIST 数据集的加载器
├── README.md # 本文件
├── data # 数据集文件夹, 供 PyTorch DataLoader 使用
├── src # Tensor 的 C++ 源代码实现（基于第二次作业），以及 PyBind 的模块定义
│   ├── CMakeLists.txt # CMake 配置文件，包括了 PyBind11 的配置和模块打包
│   ├── device_space.h
│   ├── global_curand_generator.cuh
│   ├── hw2.pdf # 第二次作业报告，解释了 C++ Tensor 部分的设计
│   ├── main.cu
│   ├── README.md # C++部分，第二次作业的设计文档
│   ├── tensor.cu
│   ├── tensor.h
│   ├── tensor_kernel.h
│   ├── tensor_module.cu # PyBind 模块定义
│   └── tensornn.cuh
└── test # 所有的 PyTest 单元测试用例
    ├── Genshin.cpython-312-x86_64-linux-gnu.so # PyBind 模块编译结果
    ├── test_conv2d.py
    ├── test_fc.py
    ├── test_loss.py
    ├── test_max_pooling.py
    ├── test_relu.py
    ├── test_sigmoid.py
    └── test_softmax.py
```

## 作业要求及解释

### 要求1: 封装 Tensor 并实例化

`import` 编译好的 PyBind模块，并使用构造函数 `Tensor.iota` 和 `Tensor.uniform` 实例化 Tensor 对象.

```python
from Genshin import Tensor, TensorDevice, nn

# 生成一个 3x3 的 Tensor 对象
test_tensor = Tensor.uniform([3, 3], TensorDevice.GPU)
```

### 要求2: 在 Python 中封装 7 种常用 Module, 并实现 Tensor 的输入输出

7 种 Module 的调用实例可见所有的单元测试 `test/test_*.py`. Tensor 可接受 NumPy 数组作为输入：

```python
numpy_array = np.random.randn(N, C).astype(np.float32)
converted_tensor = Tensor.from_numpy(numpy_array)
```

Tensor 可以使用 `print` 输出：
```python
print(converted_tensor)
```

### 要求3: 读取 MNIST，转换为 NumPy，再转换为自己的 Tensor

见 `mnist.py`，使用了 `torchvision` 来加载 MNIST 数据集.

### 要求4: 单元测试

使用 PyTest 进行了所有 Module 的单元测试，测试代码位于 `test/`，对比的基准是 PyTorch 的实现（或者 NumPy 手写的实现，对于 Loss）
