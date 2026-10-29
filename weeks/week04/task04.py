from dl_lib.engine.graph import draw_dot, trace
from dl_lib.engine.value import Value


def main() -> int:
    a = Value(-3, "a")
    b = Value(2, "b")
    c = a * b
    c.label = "e"
    d = Value(10, "c")
    e = d + c
    e.label = "d"
    f = Value(5)
    output = f * e
    output.label = "output"

    nodes, children = trace(e)
    print("output =", output)
    print("output.grad =", output.grad)
    print("d.grad =", d.grad)

    dot = draw_dot(output)

    print(dot.source)
    # dot.render("graph", format="svg", view=True)
    return 0
