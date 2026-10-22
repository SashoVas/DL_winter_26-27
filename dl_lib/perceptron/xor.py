import numpy as np

from dl_lib.activations.functional import sigmoid


class Xor:
    def __init__(self, low: float = 0, high: float = 1, seed: int | None = None):
        rng = np.random.default_rng(seed=seed)

        self.hidden_weights = rng.uniform(low, high, 4).reshape(2, 2)
        self.output_weights = rng.uniform(low, high, 2)
        self.hidden_bias = rng.uniform(low, high, 2)
        self.output_bias = rng.uniform(low, high, 1)[0]
        rng = np.random.default_rng(seed=seed)

    def forward(self, x: np.ndarray):
        hidden = sigmoid(x @ self.hidden_weights + self.hidden_bias)
        output = sigmoid(hidden @ self.output_weights + self.output_bias)
        return output


def forward(model: Xor, x1: float, x2: float):
    return model.forward(np.array([x1, x2]))
