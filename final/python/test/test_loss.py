import pytest
import numpy as np
from Designant import *


@pytest.mark.parametrize("N, C", [
    (2, 3),
    # (1, 5),
    # (40, 100)
])
def test_softmax_cross_entropy_loss(N, C):
    np.random.seed(42)

    input_np = np.random.rand(N, C).astype(np.float32)
    ground_truth = np.random.randint(0, C, N)

    def softmax_np(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)
    def cross_entropy_loss_np(softmax_output, labels):
        N = softmax_output.shape[0]
        log_likelihood = -np.log(softmax_output[np.arange(N), labels])
        return np.sum(log_likelihood) / N

    expected_loss_numpy = cross_entropy_loss_np(softmax_np(input_np), ground_truth)

    output_designant = nnn.functional.softmax_cross_entropy(Tensor.from_numpy(input_np, False), Tensor.from_numpy(ground_truth, False)).numpy()

    np.testing.assert_allclose(output_designant, expected_loss_numpy, rtol=1e-5, atol=1e-6)
#
# import pytest
# import numpy as np
# import torch
#
# @pytest.mark.parametrize("N, C", [
#     (2, 3),  # Example case with batch=2, classes=3
#     (1, 5),  # Single sample, 5 classes
#     (4, 10)  # Batch of 4, 10 classes
# ])
# def test_softmax_cross_entropy_loss(N, C):
#     np.random.seed(42)
#
#     input_np = np.random.rand(N, C).astype(np.float32)
#     ground_truth = np.random.randint(0, C, N)
#
#     def softmax_np(x):
#         e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
#         return e_x / np.sum(e_x, axis=1, keepdims=True)
#
#     def cross_entropy_loss_np(softmax_output, labels):
#         N = softmax_output.shape[0]
#         log_likelihood = -np.log(softmax_output[np.arange(N), labels])
#         return np.sum(log_likelihood) / N
#
#     def combined_backward(softmax_output, labels):
#         grad = softmax_output.copy()
#         grad[np.arange(N), labels] -= 1
#         return grad
#
#     expected_grad = combined_backward(softmax_np(input_np), ground_truth)
#
#     actual_grad = nn.backward_softmax_cross_entropy(Tensor.from_numpy(softmax_np(input_np)), Tensor.from_numpy(ground_truth)).numpy()
#
#     np.testing.assert_allclose(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)