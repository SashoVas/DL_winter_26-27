import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main(
    slope: float = 2,
    intercept: float = 1,
    num_points: int = 50,
    range: tuple[float, float] = (0, 10),
) -> None:
    matplotlib.use("Agg")

    print(f"Generated {num_points} points for y = {slope}*x + {intercept} + noise")
    x = np.linspace(range[0], range[1], num_points)
    noise = np.random.randn(num_points)
    y = slope * x + intercept + noise
    print("x range:", list(range))
    print(f"y range: [{y.min()},{y.max()}]")
    plt.scatter(x, y)
    plt.title("Linear Relationship")
    plt.xlabel("x")
    plt.ylabel("y")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
