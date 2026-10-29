from __future__ import annotations

import math


class Value:
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

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data + other.data, label=f"{self.label}+{other.label}")
        res._op = "+"
        res._prev = {self, other}

        def _backward():
            self.grad += res.grad
            other.grad += res.grad

        res._backward = _backward

        return res

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data * other.data, label=f"{self.label}*{other.label}")
        res._op = "*"
        res._prev = {self, other}

        def _backward():
            self.grad += res.grad * other.data
            other.grad += res.grad * self.data

        res._backward = _backward
        return res

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __pow__(self, other: float) -> Value:
        res = Value(self.data**other, label=f"({self.label}**{other})")
        res._op = "**"
        res._prev = {self}

        def _backward():
            self.grad += res.grad * other * (self.data ** (other - 1))

        res._backward = _backward
        return res

    def __truediv__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data / other.data, label=f"({self.label}/{other.label})")
        res._op = "/"
        res._prev = {self, other}

        def _backward():
            self.grad += res.grad * (1 / other.data)
            other.grad += res.grad * (-self.data / other.data**2)

        res._backward = _backward
        return res

    def exp(self) -> Value:
        res = Value(math.exp(self.data), label=f"(e^{self.label})")
        res._op = "exp"
        res._prev = {self}

        def _backward():
            self.grad += res.grad * res.data

        res._backward = _backward

        return res

    def tanh(self) -> Value:

        res = Value(
            (math.exp(self.data) - math.exp(-self.data))
            / (math.exp(self.data) + math.exp(-self.data)),
            label=f"tanh({self.label})",
        )
        res._op = "tanh"
        res._prev = {self}

        def _backward():
            self.grad += (1 - res.data**2) * res.grad

        res._backward = _backward
        return res

    def backward(self):
        self.grad = 1.0
        self._backward()
        for child in self._prev:
            child._backward()
