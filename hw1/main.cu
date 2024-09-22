#include "tensor.h"

int main() {
    Tensor t({2, 3, 4}, TensorDevice::CPU);

    for (int i = 0; i < t.size(); i++) {
        // Fill with random double from -1 to 1
        t.data[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    }

    Tensor g = t.gpu();
    Tensor h = g.relu();

    h.print(std::cout);

    return 0;
}