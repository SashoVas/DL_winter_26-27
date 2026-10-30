from dl_lib.engine.mlp import Layer


def main() -> int:
    layer = Layer(3, 4)
    x = [2.0, 3.0, -1]
    res = layer(x)
    print("layer(x) =", [i.data for i in res])
    print("Number of neurons:", len(res))
    return 0
