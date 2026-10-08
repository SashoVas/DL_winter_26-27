import numpy as np


def main() -> None:
    matrix = np.arange(1, 7).reshape(2, 3)
    print("Matrix + scalar (10):")
    print(matrix + 10)
    print()
    print("Matrix (2, 3) + row vector (3,):")
    print(matrix + np.arange(10, 40, 10))
    print()
    print("Matrix (2, 3) + column vector (2, 1):")
    print(matrix + np.arange(100, 300, 100).reshape(2, 1))
    print()
    print("Matrix (2, 3) + incompatible (2,) raises:")

    try:
        matrix + np.arange(2)
    except ValueError as e:
        print(str(e))


if __name__ == "__main__":
    main()
