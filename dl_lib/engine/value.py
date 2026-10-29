from __future__ import annotations


class Value:
    def __init__(self, data: float):
        self.data = data

    def __repr__(self) -> str:
        cls = self.__class__.__name__
        return f"{cls}(data={self.data})"

    def __add__(self, other: Value) -> Value:
        return Value(self.data + other.data)

    def __mul__(self, other: Value) -> Value:
        return Value(self.data * other.data)
