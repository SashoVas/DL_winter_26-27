from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    X = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    y = [1.0, -1.0, -1.0, 1.0]
    lr = 0.05

    for epoch in range(20):
        y_hat = [mlp(x) for x in X]
        se_loss = sum((y_hat_i + (-y_i)) ** 2 for y_hat_i,
                      y_i in zip(y_hat, y))
        se_loss.backward()
        print(
            f"epoch {epoch}: loss = {se_loss.data}, sum|grad| = {sum(abs(p.grad) for p in mlp.parameters())}")
        for p in mlp.parameters():
            p.data -= lr * p.grad
            p.grad = 0.0

    return 0
