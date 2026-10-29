from dl_lib.engine.value import Value


def main() -> int:
    a = Value(10)
    b = Value(-6)
    d = a + b
    print("d =", d)
    print("d._op =", d._op)
    print("d._prev =", sorted([prev.data for prev in d._prev], reverse=True))
    c = Value(2)
    d = Value(-3)
    e = c * d
    print("e =", e)
    print("e._op =", e._op)
    print("e._prev =", sorted([prev.data for prev in e._prev], reverse=True))
    return 1
