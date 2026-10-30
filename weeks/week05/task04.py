from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    x = [2.0, 3.0, -1]
    res = mlp(x)
    print("mlp(x) =", res)
    print("Number of layers:", len(mlp.layers))
    return 0
