from dl_lib.engine.value import Value


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

    lr = 0.01

    new_output = build_output(
        a.data - lr * a.grad,
        b.data - lr * b.grad,
        c.data - lr * c.grad,
        f.data - lr * f.grad,
    )
    print(f"L before step: {output.data}")
    print(f"L after step: {new_output}")
    return 0
