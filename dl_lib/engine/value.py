from __future__ import annotations


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

        res = Value(self.data + other.data)
        res._op = "+"
        res._prev = {self, other}

        return res

    def __radd__(self, other: float) -> Value:
        return self + other

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)

        res = Value(self.data * other.data)
        res._op = "*"
        res._prev = {self, other}
        return res

    def __rmul__(self, other: float) -> Value:
        return self * other
