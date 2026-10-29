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

    o = logit.tanh()
    o.label = "o"

    o.grad = 1

    logit.grad = 1 - o.data**2

    b.grad = logit.grad
    sum.grad = logit.grad

    first.grad = sum.grad
    second.grad = sum.grad

    x1.grad = first.grad * w1.data
    w1.grad = first.grad * x1.data

    x2.grad = second.grad * w2.data
    w2.grad = second.grad * x2.data

    print("x1.grad =", x1.grad)
    print("w1.grad =", w1.grad)
    print("x2.grad =", x2.grad)
    print("w2.grad =", w2.grad)
    dot = draw_dot(o)
    print(dot.source)
    # dot.render("graph", format="svg", view=True)
    return 0
