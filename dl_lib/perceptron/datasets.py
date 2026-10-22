def create_dataset(n: int) -> list[tuple]:
    return [(i, 2 * i) for i in range(n)]


def and_dataset():
    return [([i, j], i & j) for i in [0, 1] for j in [0, 1]]


def or_dataset():
    return [([i, j], i | j) for i in [0, 1] for j in [0, 1]]


def nand_dataset():
    return [([i, j], not (i and j)) for i in [0, 1] for j in [0, 1]]
