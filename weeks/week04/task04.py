from dl_lib.engine.graph import draw_dot
from dl_lib.engine.value import Value


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

    print("output =", output)
    print("output.grad =", output.grad)
    print("d.grad =", d.grad)

    dot = draw_dot(output)

    print(dot.source)
    # dot.render("graph", format="svg", view=True)
    return 0
