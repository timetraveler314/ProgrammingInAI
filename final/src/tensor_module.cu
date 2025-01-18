#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "ndarray.h"
#include "nn.cuh"

namespace py = pybind11;

py::array_t<float> numpy(const NdArray &t) {
    py::array_t<float> np(t.getShape());
    float *np_data = np.mutable_data();
    cudaMemcpy(np_data, t.getRawData(), t.size() * sizeof(float),
               t.getDevice() == Device::CPU ? cudaMemcpyHostToHost
                                                : cudaMemcpyDeviceToHost);
    return np;
}

NdArray from_numpy(py::array_t<float, py::array::c_style | py::array::forcecast> np) {
    std::vector<int> shape(np.shape(), np.shape() + np.ndim());
    // By default, we put the tensor on the GPU
    NdArray t(shape, Device::GPU);
    cudaMemcpy(t.getRawData(), np.data(), t.size() * sizeof(float),
               t.getDevice() == Device::CPU ? cudaMemcpyHostToHost
                                               : cudaMemcpyHostToDevice);
    return t;
}

PYBIND11_MODULE(Genshin, m) {
    py::enum_<Device>(m, "TensorDevice")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<NdArray>(m, "Tensor", py::buffer_protocol())
        .def("numpy", &numpy)
        .def_static("from_numpy", &from_numpy)
        .def(py::init<const std::vector<int>&, Device>(),
             py::arg("shape"), py::arg("device"))
        .def_static("iota", &NdArray::iota)
        .def_static("uniform", &NdArray::uniform)
        .def("shape", &NdArray::getShape)
        .def("__repr__", [](const NdArray &t) {
            std::string device = t.getDevice() == Device::CPU ? "CPU" : "GPU";
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
        .def("__str__", &NdArray::toString);

    // NdArrayNN namespace
    py::module nn = m.def_submodule("nn");
    nn.def("forward_relu", &NdArrayNN::forward_relu);
    nn.def("backward_relu", &NdArrayNN::backward_relu);
    nn.def("forward_sigmoid", &NdArrayNN::forward_sigmoid);
    nn.def("backward_sigmoid", &NdArrayNN::backward_sigmoid);
    nn.def("forward_fc", &NdArrayNN::forward_fc);
    nn.def("backward_fc", &NdArrayNN::backward_fc);
    nn.def("conv2d_3x3", &NdArrayNN::conv2d_3x3);
    nn.def("conv2d_3x3_backward", &NdArrayNN::conv2d_3x3_backward);
    nn.def("forward_max_pooling_2x2", &NdArrayNN::forward_max_pooling_2x2);
    nn.def("backward_max_pooling_2x2", &NdArrayNN::backward_max_pooling_2x2);
    nn.def("forward_softmax", &NdArrayNN::forward_softmax);
    nn.def("cross_entropy", &NdArrayNN::cross_entropy);
    nn.def("backward_softmax_cross_entropy", &NdArrayNN::backward_softmax_cross_entropy);
}