"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一个基本完善的Tensor类，但是缺少梯度计算的功能
你需要把梯度计算所需要的运算的正反向计算补充完整
一共有12*2处
当你填写好之后，可以调用test_task1_*****.py中的函数进行测试
"""
import numpy
import numpy as np
from typing import List, Optional, Tuple, Union
from device import cpu, Device
from basic_operator import Op, Value
from task2_autodiff import compute_gradient_of_variables


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )

    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        return np.array(numpy_array, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not tensor.requires_grad:
            return tensor.detach()
        tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data
            if not isinstance(data, Tensor)
            else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor

    @property
    def data(self):
        return self.detach()

    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (
            value.dtype,
            self.dtype,
        )
        self.cached_data = value.realize_cached_data()

    def detach(self):
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape

    @property
    def dtype(self):
        return self.realize_cached_data().dtype

    @property
    def device(self):
        return cpu()


    def backward(self, out_grad=None):
        def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
            """Generate constant Tensor"""
            device = cpu() if device is None else device
            array = device.ones(*shape, dtype=dtype) * c  # note: can change dtype
            return Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)

        def ones(*shape, device=None, dtype="float32", requires_grad=False):
            """Generate all-ones Tensor"""
            return constant(
                *shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad
            )

        out_grad = (
            out_grad
            if out_grad
            else ones(*self.shape, dtype=self.dtype, device=self.device)
        )
        compute_gradient_of_variables(self, out_grad)
        

    def __repr__(self):
        return "Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()

        return data


    def __add__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, other)
        else:
            return AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return EWiseMul()(self, other)
        else:
            return MulScalar(other)(self)

    def __pow__(self, other):
        if isinstance(other, Tensor):
            return EWisePow()(self, other)
        else:
            return PowerScalar(other)(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return EWiseAdd()(self, Negate()(other))
        else:
            return AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return EWiseDiv()(self, other)
        else:
            return DivScalar(other)(self)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def matmul(self, other):
        return MatMul()(self, other)

    def sum(self, axes=None):
        return Summation(axes)(self)

    def broadcast_to(self, shape):
        return BroadcastTo(shape)(self)

    def reshape(self, shape):
        return Reshape(shape)(self)

    def __neg__(self):
        return Negate()(self)

    def transpose(self, axes=None):
        return Transpose(axes)(self)

    __radd__ = __add__
    __rmul__ = __mul__
    __rsub__ = __sub__
    __rmatmul__ = __matmul__

class TensorOp(Op):
    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class EWiseAdd(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: np.ndarray, b: np.ndarray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: np.ndarray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """逐点乘方，用标量做指数"""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: np.ndarray) -> np.ndarray:
        return a ** self.scalar
        

    def gradient(self, out_grad, node):
        x, = node.inputs
        return self.scalar * out_grad * (x ** (self.scalar - 1))
        


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWisePow(TensorOp):
    """逐点乘方"""

    def compute(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return a**b

    def gradient(self, out_grad, node):
        if not isinstance(node.inputs[0], Tensor) or not isinstance(
            node.inputs[1], Tensor
        ):
            raise ValueError("Both inputs must be tensors.")

        a, b = node.inputs[0], node.inputs[1]
        grad_a = out_grad * b * (a ** (b - 1))
        grad_b = out_grad * (a**b) * log(a)
        return grad_a, grad_b

def power(a, b):
    return EWisePow()(a, b)


class EWiseDiv(TensorOp):
    """逐点相除"""

    def compute(self, a, b):
        return a / b
        

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad * (rhs ** (-1)), out_grad * (-lhs) * (rhs ** (-2))
        


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar
        

    def gradient(self, out_grad, node):
        return out_grad / self.scalar
        


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
        if self.axes is None:
            self.axes = [-1, -2]

    def compute(self, a):
        # if self.axes is None:
        #     return np.transpose(a)
        # return np.swapaxes(a, *self.axes)
        target = list(range(len(a.shape)))
        target[self.axes[0]] = self.axes[1]
        target[self.axes[1]] = self.axes[0]
        return np.transpose(a, target)
        

    def gradient(self, out_grad, node):
        return [out_grad.transpose(self.axes)]
        


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return np.reshape(a, self.shape)
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return [out_grad.reshape(x.shape)]
        


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return np.broadcast_to(a, self.shape)
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        grad = out_grad
        for _ in range(len(grad.shape) - len(x.shape)):
            grad = grad.sum(axes=0)
        for i, dim in enumerate(x.shape):
            if dim == 1:
                grad = grad.sum(axes=i)
            grad = Tensor(grad)

        grad = grad.reshape(x.shape)
        return [grad]


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return np.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        if self.axes is None:
            return out_grad * Tensor(np.ones_like(x))
        else:
            sum_shape = list(x.shape)
            for d in self.axes:
                sum_shape[d] = 1
            return out_grad.reshape(sum_shape).broadcast_to(x.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        # print(a.shape, b.shape)
        return np.matmul(a, b)
        

    def gradient(self, out_grad, node):
        a, b = node.inputs
        da = matmul(out_grad, b.transpose())
        db = matmul(a.transpose(), out_grad)
        # Deal with batched matmul, sum the leading dimensions
        assert len(da.shape) >= len(a.shape)
        assert len(db.shape) >= len(b.shape)
        if len(da.shape) != len(a.shape):
            da = da.sum(axes=tuple(range(len(da.shape) - len(a.shape))))
        if len(db.shape) != len(b.shape):
            db = db.sum(axes=tuple(range(len(db.shape) - len(b.shape))))
        return [da, db]
        


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a
        

    def gradient(self, out_grad, node):
        return out_grad * (-1)
        


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return np.log(a)
        

    def gradient(self, out_grad, node):
        return [out_grad / node.inputs[0]]
        


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return np.exp(a)
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        return [out_grad * exp(x)]
        


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        return np.maximum(0, a)
        

    def gradient(self, out_grad, node):
        x = node.inputs[0]
        grad = Tensor((x.realize_cached_data() > 0).astype(np.float32)) * out_grad
        return [grad]
        


def relu(a):
    return ReLU()(a)


