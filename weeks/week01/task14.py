from dl_lib.preprocessing.functional import min_max_normalize, standardize
import numpy as np


def main() -> None:
    array = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
    min_max = min_max_normalize(array)
    standardized = standardize(array)
    print("Original:    ", array)
    print("Min-max:     ", min_max)
    print("Standardized:", standardized)
    print()
    print(f"Min-max range: [{min_max.min().item():.4f}, {min_max.max().item():.4f}]")
    print(
        f"Standardized mean/std: {standardized.mean():.4f} / {standardized.std():.4f}"
    )


if __name__ == "__main__":
    main()
