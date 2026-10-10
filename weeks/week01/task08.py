import math
import time

import numpy as np


def main(vector_len: int = 100000) -> None:
    print(f"Array length: {vector_len}")

    vec = np.random.randn(vector_len)
    iteration_sum = 0
    start = time.time()
    for i in vec:
        iteration_sum += i
    end = time.time()
    iteration_time = end - start
    print(f"Loop sum: {iteration_sum} ({iteration_time:.4f}s)")
    start = time.time()
    vectorized_sum = vec.sum()
    end = time.time()
    vectorized_time = end - start
    print(f"Vectorized sum: {vectorized_sum} ({vectorized_time:.4f}s)")
    print(
        f"Vectorized speedup: {math.floor(iteration_time / vectorized_time)}x")
    print()
    print(f"Vectorized mean: {vec.mean()}")
    print(f"Vectorized std: {vec.std()}")
    print(f"Vectorized min/max: {vec.min()}/{vec.max()}")
