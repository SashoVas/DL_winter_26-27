from __future__ import annotations

import math


class Value:
    def __init__(self, data: float, label: str = ""):
        self.data = data
        self._prev = set()
        self._op = None
        self.grad = 0.0
        self.label = label

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(data={self.data})"

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data + other.data, label=f"{self.label}+{other.label}")
        res._op = "+"
        res._prev = {self, other}
        return res

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data * other.data, label=f"{self.label}*{other.label}")
        res._op = "*"
        res._prev = {self, other}

        return res

    def __rmul__(self, other: float) -> Value:
        return self * other

    def tanh(self) -> Value:

        res = Value(
            (math.exp(self.data) - math.exp(-self.data))
            / (math.exp(self.data) + math.exp(-self.data)),
            label=f"tanh({self.label})",
        )
        res._op = "tanh"
        res._prev = {self}
        return res

    def _backward(self):
        for child in self._prev:
            child.grad += self.grad

    def backward(self):
        self.grad = 1.0
        self._backward()
        for child in self._prev:
            child._backward()
