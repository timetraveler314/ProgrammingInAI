#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "tensor.h"
#include "tensornn.cuh"

namespace py = pybind11;

PYBIND11_MODULE(Genshin, m) {
    py::enum_<TensorDevice>(m, "TensorDevice")
        .value("CPU", TensorDevice::CPU)
        .value("GPU", TensorDevice::GPU)
        .export_values();

    py::class_<Tensor>(m, "Tensor")
        .def(py::init<const std::vector<int>&, TensorDevice>(),
             py::arg("shape"), py::arg("device"))
        .def_static("iota", &Tensor::iota)
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
        // .def("__add__", &operator+)
        // .def("__sub__", &operator-)
        // .def("__mul__", &operator*)
        // .def("__truediv__", &operator/);

    // TensorNN namespace
    py::module nn = m.def_submodule("nn");
    nn.def("forward_fc", &TensorNN::forward_fc);
    nn.def("backward_fc", &TensorNN::backward_fc);
}