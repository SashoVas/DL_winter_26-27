from dl_lib.engine.mlp import Neuron


def main() -> int:
    neuron = Neuron(3)
    x = [2.0, 3.0, -1]
    print("neuron.w =", [i.data for i in neuron.w])
    print("neuron.b =", neuron.b.data)
    print("neuron(x) =", neuron(x))
    return 0
