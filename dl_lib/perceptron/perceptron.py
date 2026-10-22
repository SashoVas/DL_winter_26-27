from collections.abc import Callable

import numpy as np

from dl_lib.perceptron.functional import initialize_weights


class Perceptron:
    def __init__(
        self,
        size: int,
        low: float = 0,
        high: float = 1,
        seed: int | None = None,
        use_bias: bool = False,
        activation: Callable = lambda x: x,
    ):
        self.weights = initialize_weights(low, high, size, seed)
        self.use_bias = use_bias
        self.bias = np.array(0)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        return self.activation(x @ self.weights + self.bias)
