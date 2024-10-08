#include "tensor.h"

int main() {
    // Tensor t({2, 3, 4}, TensorDevice::CPU);
    //
    // for (int i = 0; i < t.size(); i++) {
    //     // Fill with random double from -1 to 1
    //     t.data->space[i] = 2.0 * (rand() / (double)RAND_MAX) - 1.0;
    // }
    //
    // Tensor g = t.gpu();
    // Tensor h = g.sigmoid();
    //
    // std::cout << h << std::endl;

    Tensor t1({2, 3, 4}, TensorDevice::CPU);

    t1.acceptModifier([](const DeviceSpace& space) {
        for (int i = 0; i < space.size; i++) {
            space.space[i] = i + 1;
        }
    });

    std::cout << t1 << std::endl;

    Tensor t2({2, 3, 4}, TensorDevice::CPU);

    for (int i = 0; i < t2.size(); i++) {
        t2.data->space[i] = i + 2;
    }

    std::cout << t2 << std::endl;

    auto g1 = t1.gpu();
    auto g2 = t2.gpu();

    std::cout << (t1 / t2) << std::endl;

    return 0;
}