import random
from dl_lib.engine.value import Value


class Neuron:
    def __init__(self, n_inputs, rng=random.Random(42)):

        self.w = [Value(rng.uniform(-1, 1)) for i in range(n_inputs)]
        self.b = Value(rng.uniform(-1, 1))

    def __call__(self, x: list):
        logit = sum(x_i * w_i for x_i, w_i in zip(x, self.w)) + self.b
        y_hat = logit.tanh()

        return y_hat

    def parameters(self):
        return [*self.w, self.b]


class Layer:
    pass


class MLP:
    pass
