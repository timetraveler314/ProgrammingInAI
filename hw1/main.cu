#include "tensor.h"

Tensor random_gpu_tensor(const std::vector<int>& shape);
Tensor get_test_neg1_1_tensor();

int main() {
    srand(42); // seed random number generator to allow for reproducing the results.

    Tensor t1 = get_test_neg1_1_tensor();
    std::cout << "Input Tensor t1:" << std::endl;
    std::cout << t1 << std::endl << std::endl;

    Tensor t2 = t1.relu();
    std::cout << "ReLU of t1:" << std::endl;
    std::cout << t2 << std::endl << std::endl;

    Tensor t3 = t1.sigmoid();
    std::cout << "Sigmoid of t1:" << std::endl;
    std::cout << t3 << std::endl << std::endl;

    Tensor grad = random_gpu_tensor(t1.shape);
    std::cout << "Upstream gradient:" << std::endl;
    std::cout << grad << std::endl << std::endl;

    Tensor t4 = Tensor::relu_backward(t1, grad);
    std::cout << "ReLU backward of t1:" << std::endl;
    std::cout << t4 << std::endl << std::endl;

    Tensor t5 = Tensor::sigmoid_backward(t1, grad);
    std::cout << "Sigmoid backward of t1:" << std::endl;
    std::cout << t5 << std::endl << std::endl;

    return 0;
}

Tensor random_gpu_tensor(const std::vector<int>& shape) {
    Tensor t(shape, TensorDevice::CPU);

    t.acceptModifier([](const DeviceSpace& space) {
        for (int i = 0; i < space.size; i++) {
            space.space[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
        }
    });

    return t.gpu();
}

Tensor get_test_neg1_1_tensor() {
    Tensor t1({2,3,4}, TensorDevice::CPU);
    t1.acceptModifier([](const DeviceSpace& space) {
        for (int i = 0; i < space.size; i++) {
            space.space[i] = 2 * ((double) i / space.size) - 1;
        }
    });
    return t1.gpu();
}
