import pytest
import torch
import numpy as np
from Genshin import Tensor, TensorDevice, nn

@pytest.mark.parametrize("N, C, H, W, K", [
    (2, 3, 8, 8, 5),  # Example case with batch=2, channels=3, height=8, width=8, and 5 kernels
    (1, 1, 5, 5, 2),  # Small 5x5 grayscale image
    (3, 3, 160, 160, 40) # Larger case
])
def test_conv2d_3x3(N, C, H, W, K):
    torch.manual_seed(42)
    np.random.seed(42)

    # Generate random input and kernel tensors
    images_np = np.random.rand(N, C, H, W).astype(np.float32)
    kernels_np = np.random.rand(K, C, 3, 3).astype(np.float32)

    images_torch = torch.tensor(images_np, requires_grad=False)
    kernels_torch = torch.tensor(kernels_np, requires_grad=False)

    # PyTorch Reference
    conv = torch.nn.Conv2d(C, K, kernel_size=3, padding=1, bias=False)
    conv.weight.data = kernels_torch
    expected_output_torch = conv(images_torch).detach().numpy()

    images_gen = Tensor.from_numpy(images_np)
    kernels_gen = Tensor.from_numpy(kernels_np)

    output_gen = nn.conv2d_3x3(images_gen, kernels_gen).numpy()

    # Compare the outputs
    np.testing.assert_allclose(output_gen, expected_output_torch, rtol=1e-5, atol=1e-6)

