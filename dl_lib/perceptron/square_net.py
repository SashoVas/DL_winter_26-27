import numpy as np

from dl_lib.activations.functional import sigmoid


class SquareNet:
    def __init__(
        self, hidden_units=2, low: float = 0, high: float = 1, seed: int | None = None
    ):
        rng = np.random.default_rng(seed=seed)
        self.hidden_units = hidden_units
        self.hidden_weights = rng.uniform(low, high, hidden_units)
        self.output_weights = rng.uniform(low, high, hidden_units)
        self.hidden_bias = rng.uniform(low, high, hidden_units)
        self.output_bias = rng.uniform(low, high, 1)[0]
        rng = np.random.default_rng(seed=seed)

    def forward(self, x: np.ndarray):
        if x.ndim == 1:
            x = x.reshape(len(x), 1)
        hidden = sigmoid(
            x @ self.hidden_weights.reshape(1, self.hidden_units) + self.hidden_bias
        )
        output = hidden @ self.output_weights + self.output_bias
        return output
