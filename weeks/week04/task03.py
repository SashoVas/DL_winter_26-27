from dl_lib.engine.graph import draw_dot, trace
from dl_lib.engine.value import Value


def main() -> int:
    a = Value(-3, "a")
    b = Value(2, "b")
    c = a * b
    c.label = "c"
    d = Value(10, "d")
    e = d + c
    e.label = "e"

    nodes, children = trace(e)
    print("Number of nodes:", len(nodes))
    print("Number of edges:", len(children))

    dot = draw_dot(e)

    print(dot.source)
    # dot.render("graph", format="svg", view=True)
    return 0
