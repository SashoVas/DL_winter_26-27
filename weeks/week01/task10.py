import numpy as np


def main() -> None:
    A = np.arange(1, 5).reshape(2, 2)
    B = np.arange(5, 9).reshape(2, 2)
    print("A:")
    print(A)
    print()
    print("B:")
    print(B)
    print()
    print("Elementwise:")
    print(A * B)
    print()
    print("Matrix multiplication:")
    print(A @ B)
    print()
    print("The transpose of A:")
    print(A.T)


if __name__ == "__main__":
    main()
