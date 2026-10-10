import numpy as np


def main():
    matrix = np.arange(1, 13).reshape(3, 4)
    print("Matrix:")
    print(matrix)
    print()
    print("Second row:", matrix[1])
    print("Third column:", matrix[:, 2])
    print("Submatrix:")
    print(matrix[0:2, 1:3])
    print()
    print("Boolean mask for more than 6:")
    print(matrix > 6)
    print("Values satisfying the mask:", matrix[matrix > 6])
