import numpy as np


def main() -> None:
    vec = np.arange(1, 6)
    print(
        f"1D array (vector): shape={vec.shape}, dtype={vec.dtype}, ndim={vec.ndim}, size={vec.size}"
    )
    print(vec)
    print()
    matrix = np.arange(1, 7).reshape(2, 3)
    print(
        f"2D array (matrix): shape={matrix.shape}, dtype={matrix.dtype}, ndim={matrix.ndim}, size={matrix.size}"
    )
    print(matrix)
    print()
    matrix2 = np.arange(0, 24).reshape(2, 3, 4)
    print(
        f"3D array (volume): shape={matrix2.shape}, dtype={matrix2.dtype}, ndim={matrix2.ndim}, size={matrix2.size}"
    )
    print(matrix2)