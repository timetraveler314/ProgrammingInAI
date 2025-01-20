#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include "ndarray.h"
#include "nn.cuh"
#include "tensor.h"
#include "tensornn.h"

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

PYBIND11_MODULE(Designant, m) {
    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<NdArray>(m, "ndarr", py::buffer_protocol())
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
            return "<NdArray shape=" + shape + " on " + device + ">";
        })
        .def("__str__", &NdArray::toString);

    // NdArrayNN namespace
    py::module ndarr_nn = m.def_submodule("ndarr.nn");
    ndarr_nn.def("forward_relu", &NdArrayNN::forward_relu);
    ndarr_nn.def("backward_relu", &NdArrayNN::backward_relu);
    ndarr_nn.def("forward_sigmoid", &NdArrayNN::forward_sigmoid);
    ndarr_nn.def("backward_sigmoid", &NdArrayNN::backward_sigmoid);
    ndarr_nn.def("forward_fc", &NdArrayNN::forward_fc);
    ndarr_nn.def("backward_fc", &NdArrayNN::backward_fc);
    // ndarr_nn.def("conv2d_3x3", &NdArrayNN::conv2d3, 1, 1>);
    // ndarr_nn.def("conv2d_3x3_backward", &NdArrayNN::conv2d_backward<3, 1, 1>);
    ndarr_nn.def("forward_max_pooling_2x2", &NdArrayNN::forward_max_pooling_2x2);
    ndarr_nn.def("backward_max_pooling_2x2", &NdArrayNN::backward_max_pooling_2x2);
    ndarr_nn.def("forward_softmax", &NdArrayNN::forward_softmax);
    ndarr_nn.def("cross_entropy", &NdArrayNN::cross_entropy);
    ndarr_nn.def("backward_softmax_cross_entropy", &NdArrayNN::backward_softmax_cross_entropy);

    py::class_<Tensor>(m, "Tensor", py::buffer_protocol())
        .def("numpy", [](const Tensor& t) {
            return numpy(t.to_value()->realize());
        })
        .def_static("from_numpy", [](py::array_t<float, py::array::c_style | py::array::forcecast> np, bool requires_grad) {
            return Tensor(from_numpy(np), requires_grad);
        })
        .def_static("xavier", &Tensor::xavier)
        .def_static("zeros_like", &Tensor::zeros_like)
        .def("shape", &Tensor::getShape)
        .def("__repr__", [](const Tensor &t) {
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
        .def("__str__", &Tensor::toString)
        // Operators
        .def("__add__", &Tensor::operator+)
        .def("__sub__", static_cast<Tensor (Tensor::*)(const Tensor&) const>(&Tensor::operator-))
        .def("__neg__", static_cast<Tensor (Tensor::*)() const>(&Tensor::operator-))
        .def(py::self + float())
        .def(float() * py::self)
        .def(py::self / float())
        .def(py::self * py::self)
        .def(py::self / py::self)
        .def("__pow__", &Tensor::operator^)
        .def("__matmul__", &Tensor::operator%)
        .def("transpose", &Tensor::transpose)
        .def("reshape", &Tensor::reshape)
        .def("backward", &Tensor::backward)
        .def("grad", &Tensor::grad)
        .def("detach", &Tensor::detach)
        .def("update", &Tensor::update);

    py::module nn = m.def_submodule("nnn");
    py::class_<TensorNN::Conv2D_3x3>(nn, "Conv2d_3x3")
        .def(py::init<int, int>(), py::arg("in_channels"), py::arg("out_channels"))
        .def_readonly("in_channels", &TensorNN::Conv2D_3x3::C)
        .def_readonly("out_channels", &TensorNN::Conv2D_3x3::K)
        .def_readwrite("kernels", &TensorNN::Conv2D_3x3::kernels)
        .def("__call__", &TensorNN::Conv2D_3x3::operator());

    py::class_<TensorNN::Conv2D>(nn, "Conv2d")
        .def(py::init<int, int, int, int, int>(), py::arg("in_channels"), py::arg("out_channels"),
            py::arg("kernel_size"), py::arg("stride"), py::arg("padding"))
        .def_readonly("in_channels", &TensorNN::Conv2D::C)
        .def_readonly("out_channels", &TensorNN::Conv2D::K)
        .def_readwrite("kernels", &TensorNN::Conv2D::kernels)
        .def("__call__", &TensorNN::Conv2D::operator());

    py::class_<TensorNN::Linear>(nn, "Linear")
        .def(py::init<int, int, int>(), py::arg("batch_size"), py::arg("in_features"), py::arg("out_features"))
        .def_readonly("batch_size", &TensorNN::Linear::batch_size)
        .def_readonly("in_features", &TensorNN::Linear::in_features)
        .def_readonly("out_features", &TensorNN::Linear::out_features)
        .def_readwrite("weight", &TensorNN::Linear::weight)
        .def_readwrite("bias", &TensorNN::Linear::bias)
        .def("__call__", &TensorNN::Linear::operator());

    py::module nn_functional = nn.def_submodule("functional");
    nn_functional.def("relu", &TensorFunctional::ReLU);
    nn_functional.def("sigmoid", &TensorFunctional::Sigmoid);
    nn_functional.def("softmax_cross_entropy", &TensorFunctional::SoftmaxCrossEntropy);
    nn_functional.def("maxpool2d", &TensorFunctional::MaxPool2D);
}