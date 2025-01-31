"""
此次作业借鉴和参考了Needle项目 https://github.com/dlsyscourse/lecture5
本文件我们给出一些可能会用到的检验task1梯度计算部分的函数
"""

import numpy as np
from task1_operators import *

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-6
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [
            x.numpy()
            for x in out.op.gradient_as_tuple(Tensor(np.ones(out.shape)), out)
        ]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i]) for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_power_scalar_backward():
    gradient_check(
        power_scalar, Tensor(np.random.randn(5, 4)), scalar=np.random.randint(1)
    )

def test_divide_backward():
    gradient_check(
        divide,
        Tensor(np.random.randn(5, 4)),
        Tensor(5 + np.random.randn(5, 4)),
    )


def test_divide_scalar_backward():
    gradient_check(
        divide_scalar, Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1)
    )


def test_matmul_simple_backward():
    gradient_check(
        matmul, Tensor(np.random.randn(5, 4)), Tensor(np.random.randn(4, 5))
    )


def test_matmul_batched_backward():
    gradient_check(
        matmul,
        Tensor(np.random.randn(6, 6, 5, 4)),
        Tensor(np.random.randn(6, 6, 4, 3)),
    )
    gradient_check(
        matmul,
        Tensor(np.random.randn(6, 6, 5, 4)),
        Tensor(np.random.randn(4, 3)),
    )
    gradient_check(
        matmul,
        Tensor(np.random.randn(5, 4)),
        Tensor(np.random.randn(6, 6, 4, 3)),
    )


def test_reshape_backward():
    gradient_check(reshape, Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(negate, Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(transpose, Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(transpose, Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))


def test_broadcast_to_backward():
    gradient_check(broadcast_to, Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(broadcast_to, Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(
        broadcast_to,
        Tensor(
            np.random.randn(
                1,
            )
        ),
        shape=(3, 3, 3),
    )
    gradient_check(broadcast_to, Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(
        broadcast_to, Tensor(np.random.randn(5, 4, 1)), shape=(5, 4, 3)
    )


def test_summation_backward():
    gradient_check(summation, Tensor(np.random.randn(5, 4)), axes=(1,))
    gradient_check(summation, Tensor(np.random.randn(5, 4)), axes=(0,))
    gradient_check(summation, Tensor(np.random.randn(5, 4)), axes=(0, 1))
    gradient_check(summation, Tensor(np.random.randn(5, 4, 1)), axes=(0, 1))



if __name__ == "__main__":
    ## 可以分别测试每个函数
    # test_power_scalar_backward()
    # test_ewisepow_backward()
    # test_divide_backward()
    # test_divide_scalar_backward()
    # test_matmul_backward()
    # test_summation_backward()
    # test_broadcast_to_backward()
    # test_reshape_backward()
    # test_negate_backward()
    # test_transpose_backward()
    ## log 和 exp 的测试没写...
    ## 交作业的时候也是会测试的...

    # Test ReLU for myself
    gradient_check(relu, Tensor(np.random.randn(5, 10)))