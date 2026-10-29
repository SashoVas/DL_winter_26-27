from dl_lib.engine.value import Value


def manual_der(output):
    epsilon = 0.0001
    a_grad = (build_output(a_val=2 + epsilon) - output.data) / epsilon
    b_grad = (build_output(b_val=-3 + epsilon) - output.data) / epsilon
    c_grad = (build_output(c_val=10 + epsilon) - output.data) / epsilon
    f_grad = (build_output(f_val=5 + epsilon) - output.data) / epsilon

    return a_grad, b_grad, c_grad, f_grad


def build_output(a_val=2, b_val=-3, c_val=10, f_val=5) -> float:
    a = Value(a_val, "a")
    b = Value(b_val, "b")
    e = a * b
    e.label = "e"
    c = Value(c_val, "c")
    d = e + c
    d.label = "d"
    f = Value(f_val, "f")
    output = f * d
    output.label = "output"
    return output.data


def main() -> int:

    a = Value(2, "a")
    b = Value(-3, "b")
    e = a * b
    e.label = "e"
    c = Value(10, "c")
    d = e + c
    d.label = "d"
    f = Value(5, "f")
    output = f * d
    output.label = "output"

    output.grad = 1

    d.grad = f.data
    f.grad = d.data

    e.grad = d.grad
    c.grad = d.grad

    a.grad = e.grad * b.data
    b.grad = e.grad * a.data

    a_grad, b_grad, c_grad, f_grad = manual_der(output)
    print(f"Manual gradients: a={a.grad}, b={b.grad}, c={c.grad}, f={f.grad}")
    print(f"Numerical gradients: a={a_grad}, b={b_grad}, c={c_grad}, f={f_grad}")
    return 0
