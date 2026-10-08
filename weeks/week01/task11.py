import numpy as np
import matplotlib.pyplot as plt


def main(
    slope: float = 2,
    intercept: float = 1,
    num_points: int = 50,
    range: tuple[float, float] = (0, 10),
) -> None:
    print(f"Generated {num_points} points for y = {slope}*x + {intercept} + noise")
    x = np.linspace(range[0], range[1], num_points)
    noise = np.random.randn(num_points)
    y = slope * x + intercept + noise
    print("x range:", list(range))
    print(f"y range: [{y.min()},{y.max()}]")
    plt.scatter(x, y)
    plt.show()


if __name__ == "__main__":
    main()
