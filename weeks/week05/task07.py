from dl_lib.engine.mlp import MLP
from dl_lib.engine.value import Value


def calc_y_hat(X: list[list], mlp: MLP) -> list[Value]:
    y_hat: list[Value] = []
    for x in X:
        prediction = mlp(x)

        if not isinstance(prediction, Value):
            raise TypeError("Expected a single output from the MLP")

        y_hat.append(prediction)
    return y_hat


def calc_loss(y_hat: list, y: list) -> Value:
    return sum(
        ((y_hat_i + (-y_i)) ** 2 for y_hat_i, y_i in zip(y_hat, y)), start=Value(0.0)
    )


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    X = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    y = [1.0, -1.0, -1.0, 1.0]
    lr = 0.05

    for epoch in range(20):
        y_hat = calc_y_hat(X, mlp)
        se_loss = calc_loss(y_hat, y)
        se_loss.backward()
        for p in mlp.parameters():
            p.data -= lr * p.grad
            # p.grad = 0.0
        print(
            f"epoch {epoch}: loss = {se_loss.data}, sum|grad| = {sum(abs(p.grad) for p in mlp.parameters())}"
        )

    return 0
