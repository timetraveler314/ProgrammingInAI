import pytest
import numpy as np
from Designant import *


@pytest.mark.parametrize("N, C", [
    (2, 3),
    (1, 5),
    (40, 100)
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

    def softmax_loss(Z, y):
        """
        一个写了很多遍的Softmax loss...

        Args:
            Z : 2D numpy array of shape (batch_size, num_classes),
            containing the logit predictions for each class.
            y : 1D numpy array of shape (batch_size, )
                containing the true label of each example.

        Returns:
            Average softmax loss over the sample.
        """
        # Normalize
        Z -= np.max(Z, axis=1, keepdims=True)

        Z_exp = np.exp(Z)
        Z_sum = np.sum(Z_exp, axis=1, keepdims=True)
        losses = np.log(Z_sum + 1e-8) - Z[np.arange(Z.shape[0]), y]
        avg_loss = np.mean(losses)

        return avg_loss

    expected_loss_numpy = softmax_loss(input_np, ground_truth)

    output_designant = nnn.functional.softmax_cross_entropy(Tensor.from_numpy(input_np, False), Tensor.from_numpy(ground_truth, False)).numpy()

    np.testing.assert_allclose(output_designant, expected_loss_numpy, rtol=1e-5, atol=1e-6)

import pytest
import numpy as np
import torch

@pytest.mark.parametrize("N, C", [
    (2, 3),  # Example case with batch=2, classes=3
    (1, 5),  # Single sample, 5 classes
    (4, 10)  # Batch of 4, 10 classes
])
def test_back_softmax_cross_entropy_loss(N, C):
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

    def combined_backward(softmax_output, labels):
        grad = softmax_output.copy()
        grad[np.arange(N), labels] -= 1
        return grad

    expected_grad = combined_backward(softmax_np(input_np), ground_truth)

    input_tensor = Tensor.from_numpy(input_np, True)
    ground_truth_tensor = Tensor.from_numpy(ground_truth, False)
    softmax_tensor = nnn.functional.softmax_cross_entropy(input_tensor, ground_truth_tensor)

    softmax_tensor.backward()

    actual_grad = input_tensor.grad().numpy()

    np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-6)