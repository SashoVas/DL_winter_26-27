import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def main(histogram_samples_count: int = 1000) -> None:
    matplotlib.use("Agg")
    x = np.linspace(0, 2 * np.pi, 500)
    samples = np.random.randn(histogram_samples_count)
    sin_x = np.sin(x)
    cos_x = np.cos(x)
    print(f"sin(x) range:[{sin_x.min()},{sin_x.max()}]")
    print(f"Histogram sample count: {histogram_samples_count}")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(x, sin_x, label="sin(x)")
    ax1.plot(x, cos_x, label="cos(x)")
    ax1.set_title("sin(x) and cos(x)")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True)

    ax2.hist(samples, bins=30)
    ax2.set_title("Histogram")
    ax2.set_xlabel("value")
    ax2.set_ylabel("count")
    ax2.grid(True)

    fig.tight_layout()
    plt.show()
