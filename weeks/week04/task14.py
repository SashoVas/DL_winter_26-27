from dl_lib.engine.value import Value


def decomposed():
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

    o = (logit.exp() + -1 * (-1 * logit).exp()) / (logit.exp() + (-1 * logit).exp())
    o.label = "o"
    o.backward()
    print("o (decomposed) =", o)
    print("x1.grad (decomposed) =", x1.grad)
    print("w1.grad (decomposed) =", w1.grad)


def build_in():
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
    o.backward()
    print("o (built-in tanh) =", o)
    print("x1.grad (built-in tanh) =", x1.grad)
    print("w1.grad (built-in tanh) =", w1.grad)


def main() -> int:
    decomposed()
    build_in()
    return 0
