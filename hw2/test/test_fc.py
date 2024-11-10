import pytest
import torch
import numpy as np
from Genshin import Tensor, TensorDevice
from Genshin.nn import forward_fc, backward_fc

@pytest.mark.parametrize("batch_size, input_size, output_size", [
    (4, 5, 3),
    (8, 10, 6),
    (16, 20, 8)
])
def test_forward_fc(batch_size, input_size, output_size):
    input_data = np.random.rand(batch_size, input_size).astype(np.float32)
    weight_data = np.random.rand(output_size, input_size).astype(np.float32)
    bias_data = np.random.rand(output_size).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)
    weight_tensor = Tensor.from_numpy(weight_data)
    bias_tensor = Tensor.from_numpy(bias_data)

    actual_output = forward_fc(input_tensor, weight_tensor, bias_tensor).numpy()

    torch_input = torch.tensor(input_data, requires_grad=False)
    torch_weight = torch.tensor(weight_data, requires_grad=False)
    torch_bias = torch.tensor(bias_data, requires_grad=False)
    torch_fc_output = torch.nn.functional.linear(torch_input, torch_weight, torch_bias).detach().numpy()

    np.testing.assert_allclose(actual_output, torch_fc_output, rtol=1e-5, atol=1e-5, err_msg="forward_fc output mismatch")


@pytest.mark.parametrize("batch_size, input_size, output_size", [
    (4, 5, 3),
    (8, 10, 6),
    (16, 20, 8)
])
def test_backward_fc(batch_size, input_size, output_size):
    input_data = np.random.rand(batch_size, input_size).astype(np.float32)
    weight_data = np.random.rand(output_size, input_size).astype(np.float32)
    bias_data = np.random.rand(output_size).astype(np.float32)
    grad_output_data = np.random.rand(batch_size, output_size).astype(np.float32)

    input_tensor = Tensor.from_numpy(input_data)
    weight_tensor = Tensor.from_numpy(weight_data)
    bias_tensor = Tensor.from_numpy(bias_data)
    grad_output_tensor = Tensor.from_numpy(grad_output_data)

    grad_input, grad_weight, grad_bias = backward_fc(grad_output_tensor, input_tensor, weight_tensor)
    actual_grad_input = grad_input.numpy()
    actual_grad_weight = grad_weight.numpy()
    actual_grad_bias = grad_bias.numpy()

    torch_input = torch.tensor(input_data, requires_grad=True)
    torch_weight = torch.tensor(weight_data, requires_grad=True)
    torch_bias = torch.tensor(bias_data, requires_grad=True)
    torch_fc_output = torch.nn.functional.linear(torch_input, torch_weight, torch_bias)
    torch_fc_output.backward(torch.tensor(grad_output_data))

    np.testing.assert_allclose(actual_grad_input, torch_input.grad.numpy(), rtol=1e-5, atol=1e-5, err_msg="input gradient mismatch")
    np.testing.assert_allclose(actual_grad_weight, torch_weight.grad.numpy(), rtol=1e-5, atol=1e-5, err_msg="weight gradient mismatch")
    np.testing.assert_allclose(actual_grad_bias, torch_bias.grad.numpy(), rtol=1e-5, atol=1e-5, err_msg="bias gradient mismatch")
