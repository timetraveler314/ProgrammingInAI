# 人工智能中的编程 第二次作业

实验报告请见 [report.pdf](report.pdf).

由于已经实现 PyBind 封装，大部分随机化的测试都可以在 Python 中进行，因此 `main.cu` 中的输入一般是较小且有规律的 (`Tensor::iota`). 但是仍然完成了作业要求：使用 `cuRAND` 随机化网络参数. 这一点可以在 `main.cu` 的 `=== Test 2 : Convolutional layer ===` 中随机初始化卷积核来验证.

本作业在 Arch Linux 系统上使用 CUDA 12.7 和 CMake 3.31.0 进行开发，基于 `C++ 20` 标准, 沿用了第一次作业的大部分 `Tensor` 类的设计. 项目的目录结构如下：

- `test/`: 测试代码，采用 `PyTest` 进行了全部五个部分的单元测试，基准是 `PyTorch` 或者 `NumPy` 的实现.
- `main.cu`: 主程序，实现了 `main` 函数，用于测试 `Tensor` 类的基本功能.
- `tensor_module.cu`: 使用 PyBind11 封装 `Tensor` 类，使其可以在 Python 中调用.
- `CMakeLists.txt`: 项目构建文件.
- `global_curand_generator.cuh`: 随机数生成器的封装：这是一个单例模式的随机数生成器，用于给出一个全局唯一的随机数生成器，以便结果的可重现性. 其中使用了 `curand` 库来生成随机数.