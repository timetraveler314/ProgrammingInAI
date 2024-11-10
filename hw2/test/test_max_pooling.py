import pytest
import torch
import numpy as np
from Genshin import Tensor, TensorDevice, nn

@pytest.mark.parametrize("N, C, H, W", [
    (2, 3, 8, 8),
    (1, 1, 5, 5),
    (3, 3, 10, 10)
])
def test_max_pooling(N, C, H, W):
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate random input tensor (N, C, H, W)
    images_np = np.random.rand(N, C, H, W).astype(np.float32)
    images_torch = torch.tensor(images_np, requires_grad=False)

    # PyTorch reference for max pooling
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    expected_output_torch = pool(images_torch).detach().numpy()

    # Genshin custom max pooling implementation
    images_gen = Tensor.from_numpy(images_np)
    output_gen = nn.forward_max_pooling_2x2(images_gen).numpy()

    # Compare the outputs
    np.testing.assert_allclose(output_gen, expected_output_torch, rtol=1e-5, atol=1e-6)

@pytest.mark.parametrize("N, C, H, W", [
    (2, 3, 8, 8),
    (1, 1, 5, 5),
    (3, 3, 10, 10)
])
def test_max_pooling_backward(N, C, H, W):
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate random input and output gradient tensors
    images_np = np.random.rand(N, C, H, W).astype(np.float32)
    output_grad_np = np.random.rand(N, C, H // 2, W // 2).astype(np.float32)

    images_torch = torch.tensor(images_np, requires_grad=True)
    output_grad_torch = torch.tensor(output_grad_np, requires_grad=False)

    # PyTorch reference for max pooling with backward pass
    pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
    output_torch = pool(images_torch)
    output_torch.backward(output_grad_torch)

    expected_input_grad_torch = images_torch.grad.detach().numpy()

    images_gen = Tensor.from_numpy(images_np)
    output_grad_gen = Tensor.from_numpy(output_grad_np)

    input_grad_gen = nn.backward_max_pooling_2x2(output_grad_gen, images_gen)

    # Compare the gradients
    np.testing.assert_allclose(input_grad_gen.numpy(), expected_input_grad_torch, rtol=1e-5, atol=1e-6)
