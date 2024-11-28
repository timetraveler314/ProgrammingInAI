"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件是结合task1_operators.py和task2_autodiff.py的代码
该TensorFull类可以实现自动微分，你可以使用类似的结构作为Project-Part3的框架
"""

from task1_operators import Tensor
from task2_autodiff import compute_gradient_of_variables
from utils import ones

class TensorFull(Tensor):
    def __init__(
        self,
        array,
        *,
        device=None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        super().__init__(
            array,
            device=device,
            dtype=dtype,
            requires_grad=True,
            **kwargs
        )

    def backward(self, out_grad=None):
        out_grad = (
                    out_grad
                    if out_grad
                    else ones(*self.shape, dtype=self.dtype, device=self.device)
                )
        compute_gradient_of_variables(self, out_grad)