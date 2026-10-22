from collections.abc import Callable

import numpy as np

from dl_lib.perceptron.functional import forward, initialize_weights


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
        rng = np.random.default_rng(seed=seed)
        if use_bias:
            self.bias = rng.uniform(low=low, high=high, size=1)
        else:
            self.bias = np.array(0)
        self.activation = activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        return forward(x, self.weights, self.bias, self.activation)
