from dl_lib.engine.value import Value
from dl_lib.engine.graph import draw_dot


def main() -> int:

    x1 = Value(2.0, "x1")
    x2 = Value(0.0, "x2")
    w1 = Value(-3.0, "w1")
    w2 = Value(1.0, "w2")
    b = Value(6.8813735870195432, "b")
    first = x1 * w1
    first.label = "x1*w1"
    second = x2 * w2
    second.label = "x2*w2"

    sum = first + second
    sum.label = f"{first.label} + {second.label}"

    logit = sum + b
    logit.label = "logit"
    print("logit =", logit)
    dot = draw_dot(logit)
    print(dot.source)
    # dot.render("graph", format="svg", view=True)
    return 0
