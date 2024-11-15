import pytest
import numpy as np
from Genshin import Tensor, nn

@pytest.mark.parametrize("N, C", [
    (2, 3),
    (1, 5),
    (40, 100)
])
def test_softmax_numpy(N, C):
    np.random.seed(42)

    input_np = np.random.rand(N, C).astype(np.float32)

    def softmax_np(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    expected_output_numpy = softmax_np(input_np)

    input_gen = Tensor.from_numpy(input_np)
    output_gen = nn.forward_softmax(input_gen).numpy()

    np.testing.assert_allclose(output_gen, expected_output_numpy, rtol=1e-5, atol=1e-6)
