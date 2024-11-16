import pytest
import torch
import numpy as np
from Genshin import Tensor
from Genshin.nn import forward_relu, backward_relu

@pytest.mark.parametrize("N, C", [
    (4, 5),
    (8, 10),
    (16, 20)
])
def test_forward_relu(N, C):
    input_data = np.random.randn(N, C).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)

    actual_output = forward_relu(input_tensor).numpy()

    torch_input = torch.tensor(input_data, requires_grad=True)
    torch_relu_output = torch.nn.functional.relu(torch_input).detach().numpy()

    np.testing.assert_allclose(actual_output, torch_relu_output, rtol=1e-5, atol=1e-5, err_msg="ReLU forward output mismatch")


@pytest.mark.parametrize("N, C", [
    (4, 5),
    (8, 10),
    (16, 20)
])
def test_backward_relu(N, C):
    input_data = np.random.randn(N, C).astype(np.float32)
    grad_output_data = np.random.randn(N, C).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)
    grad_output_tensor = Tensor.from_numpy(grad_output_data)

    actual_grad_input = backward_relu(input_tensor, grad_output_tensor).numpy()

    torch_input = torch.tensor(input_data, requires_grad=True)
    torch_relu_output = torch.nn.functional.relu(torch_input)
    torch_relu_output.backward(torch.tensor(grad_output_data))
    torch_grad_input = torch_input.grad.numpy()

    np.testing.assert_allclose(actual_grad_input, torch_grad_input, rtol=1e-5, atol=1e-5, err_msg="ReLU backward gradient mismatch")
