#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "tensornn.cuh"

namespace py = pybind11;

py::array_t<float> numpy(const Tensor &t) {
    py::array_t<float> np(t.getShape());
    float *np_data = np.mutable_data();
    cudaMemcpy(np_data, t.getRawData(), t.size() * sizeof(float),
               t.getDevice() == TensorDevice::CPU ? cudaMemcpyHostToHost
                                                : cudaMemcpyDeviceToHost);
    return np;
}

Tensor from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> np) {
    std::vector<int> shape(np.shape(), np.shape() + np.ndim());
    // By default, we put the tensor on the GPU
    Tensor t(shape, TensorDevice::GPU);
    cudaMemcpy(t.getRawData(), np.data(), t.size() * sizeof(float),
               t.getDevice() == TensorDevice::CPU ? cudaMemcpyHostToHost
                                               : cudaMemcpyHostToDevice);
    return t;
}

PYBIND11_MODULE(Genshin, m) {
    py::enum_<TensorDevice>(m, "TensorDevice")
        .value("CPU", TensorDevice::CPU)
        .value("GPU", TensorDevice::GPU)
        .export_values();

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def("numpy", &numpy)
        .def_static("from_numpy", &from_numpy)
        .def(py::init<const std::vector<int>&, TensorDevice>(),
             py::arg("shape"), py::arg("device"))
        .def_static("iota", &Tensor::iota)
        .def_static("uniform", &Tensor::uniform)
        .def("shape", &Tensor::getShape)
        .def("__repr__", [](const Tensor &t) {
            std::string device = t.getDevice() == TensorDevice::CPU ? "CPU" : "GPU";
            std::string shape = "[";
            for (int i = 0; i < t.getShape().size(); i++) {
                shape += std::to_string(t.getShape()[i]);
                if (i != t.getShape().size() - 1) {
                    shape += ", ";
                }
            }
            shape += "]";
            return "<Tensor shape=" + shape + " on " + device + ">";
        })
        .def("__str__", &Tensor::toString);

    // TensorNN namespace
    py::module nn = m.def_submodule("nn");
    nn.def("forward_relu", &TensorNN::forward_relu);
    nn.def("backward_relu", &TensorNN::backward_relu);
    nn.def("forward_sigmoid", &TensorNN::forward_sigmoid);
    nn.def("backward_sigmoid", &TensorNN::backward_sigmoid);
    nn.def("forward_fc", &TensorNN::forward_fc);
    nn.def("backward_fc", &TensorNN::backward_fc);
    nn.def("conv2d_3x3", &TensorNN::conv2d_3x3);
    nn.def("conv2d_3x3_backward", &TensorNN::conv2d_3x3_backward);
    nn.def("forward_max_pooling_2x2", &TensorNN::forward_max_pooling_2x2);
    nn.def("backward_max_pooling_2x2", &TensorNN::backward_max_pooling_2x2);
    nn.def("forward_softmax", &TensorNN::forward_softmax);
    nn.def("cross_entropy", &TensorNN::cross_entropy);
    nn.def("backward_softmax_cross_entropy", &TensorNN::backward_softmax_cross_entropy);
}