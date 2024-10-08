# 人工智能中的编程 第一次作业

## 第一部分：

### `cifar.py`: 主程序

## 第二部分：实现 ```Tensor``` 类

本项目由 CMake 构建，使用 CUDA 12.6 和 C++20 标准.

### ```main.cu```: 主程序

### ```tensor.cu```: ```Tensor``` 类的实现： 
- 对设备内存块的抽象 `DeviceSpace`，以及其引用计数模型 `device_ptr`，用于内部原始数据的存储.
- `TensorKernel` 命名空间：定义了一系列的 CUDA 核函数，用于实现 `Tensor` 的各种计算.
- `Tensor::cpu()`, `Tensor::gpu()`：用于在 CPU 和 GPU 之间进行 `Tensor` 的拷贝.
- `Tensor::relu()`, `Tensor::sigmoid()`: 计算 `Tensor` 的逐点 ReLU 和 Sigmoid 函数并返回结果 `Tensor`.