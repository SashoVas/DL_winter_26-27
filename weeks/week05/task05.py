from dl_lib.engine.mlp import MLP


def main() -> int:
    mlp = MLP(3, [4, 4, 1])
    params = mlp.parameters()
    print("Number of parameters:", len(params))
    print("First three parameters:", [i.data for i in params[:3]])
    return 0
