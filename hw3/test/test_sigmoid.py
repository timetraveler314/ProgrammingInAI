import pytest
import torch
import numpy as np
from Genshin import Tensor
from Genshin.nn import forward_sigmoid, backward_sigmoid

@pytest.mark.parametrize("N, C", [
    (4, 5),
    (8, 10),
    (16, 20)
])
def test_forward_sigmoid(N, C):
    input_data = np.random.randn(N, C).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)

    actual_output = forward_sigmoid(input_tensor).numpy()

    torch_input = torch.tensor(input_data, requires_grad=True)
    torch_sigmoid_output = torch.sigmoid(torch_input).detach().numpy()

    np.testing.assert_allclose(actual_output, torch_sigmoid_output, rtol=1e-5, atol=1e-5, err_msg="Sigmoid forward output mismatch")


@pytest.mark.parametrize("N, C", [
    (4, 5),
    (8, 10),
    (16, 20)
])
def test_backward_sigmoid(N, C):
    input_data = np.random.randn(N, C).astype(np.float32)
    grad_output_data = np.random.randn(N, C).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)
    grad_output_tensor = Tensor.from_numpy(grad_output_data)

    actual_sigmoid_output = forward_sigmoid(input_tensor)
    actual_grad_input = backward_sigmoid(actual_sigmoid_output, grad_output_tensor).numpy()

    torch_input = torch.tensor(input_data, requires_grad=True)
    torch_sigmoid_output = torch.sigmoid(torch_input)
    torch_sigmoid_output.backward(torch.tensor(grad_output_data))
    torch_grad_input = torch_input.grad.numpy()

    np.testing.assert_allclose(actual_grad_input, torch_grad_input, rtol=1e-5, atol=1e-5, err_msg="Sigmoid backward gradient mismatch")
