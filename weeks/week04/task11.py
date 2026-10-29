from __future__ import annotations

import math


class OverwritingValue:
    def __init__(self, data: float, label: str = ""):
        self.data = data
        self._prev = set()
        self._op = None
        self.grad = 0.0
        self.label = label
        self._backward = lambda: None

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(data={self.data})"

    def __add__(self, other: OverwritingValue | float) -> OverwritingValue:
        other = (
            other if isinstance(other, OverwritingValue) else OverwritingValue(other)
        )

        res = OverwritingValue(
            self.data + other.data, label=f"{self.label}+{other.label}"
        )
        res._op = "+"
        res._prev = {self, other}

        def _backward():
            self.grad = res.grad
            other.grad = res.grad

        res._backward = _backward
        return res

    def __radd__(self, other: float) -> OverwritingValue:
        return self + other

    def __mul__(self, other: OverwritingValue | float) -> OverwritingValue:
        other = (
            other if isinstance(other, OverwritingValue) else OverwritingValue(other)
        )

        res = OverwritingValue(
            self.data * other.data, label=f"{self.label}*{other.label}"
        )
        res._op = "*"
        res._prev = {self, other}

        return res

    def __rmul__(self, other: float) -> OverwritingValue:
        return self * other

    def tanh(self) -> OverwritingValue:

        res = OverwritingValue(
            (math.exp(self.data) - math.exp(-self.data))
            / (math.exp(self.data) + math.exp(-self.data)),
            label=f"tanh({self.label})",
        )
        res._op = "tanh"
        res._prev = {self}
        return res

    def backward(self):
        self.grad = 1.0
        self._backward()
        for child in self._prev:
            child._backward()


def main() -> int:

    a = OverwritingValue(3.0, "a")
    b = a + a
    b.backward()
    print("b.data =", b.data)
    print("a.grad =", a.grad)

    return 0
