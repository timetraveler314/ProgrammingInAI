#include "tensor.h"
#include <thrust/functional.h>

Tensor random_gpu_tensor(const std::vector<int>& shape);
Tensor get_test_neg1_1_tensor();

int main() {
    srand(42); // seed random number generator to allow for reproducing the results.

    Tensor t1 = get_test_neg1_1_tensor();
    std::cout << "Input Tensor t1:" << std::endl;
    std::cout << t1 << std::endl << std::endl;

    auto t2 = t1.gpu();
    thrust::negate<TensorDataType> op;
    std::cout << t2.transform(op) << std::endl;

    return 0;
}

Tensor random_gpu_tensor(const std::vector<int>& shape) {
    Tensor t(shape, TensorDevice::CPU);


    return t.gpu();
}

Tensor get_test_neg1_1_tensor() {
    Tensor t1({2,3,4}, TensorDevice::CPU);

    return t1.gpu();
}
