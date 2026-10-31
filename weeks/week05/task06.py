from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    X = [[2.0, 3.0, -1], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]
    y = [1.0, -1.0, -1.0, 1.0]
    y_hat = [mlp(x) for x in X]
    print("Predictions:", [i.data for i in y_hat])
    se_loss = sum((y_hat_i + (-y_i)) ** 2 for y_hat_i,
                  y_i in zip(y_hat, y))  # / len(y)
    print("SE loss:", se_loss.data)
    return 0
