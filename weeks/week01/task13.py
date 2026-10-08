import numpy as np
import matplotlib.pyplot as plt


def main(samples: int = 10_000) -> None:
    uniform = np.random.rand(samples)
    normal = np.random.randn(samples)
    exponential = np.random.exponential(size=samples)
    print(
        f"Uniform: mean={uniform.mean()}, std={uniform.std()}, min={uniform.min()}, max={uniform.max()}"
    )
    print(
        f"Normal: mean={normal.mean()}, std={normal.std()}, min={normal.min()}, max={normal.max()}"
    )
    print(
        f"Exponential: mean={exponential.mean()}, std={exponential.std()}, min={exponential.min()}, max={exponential.max()}"
    )

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    for ax, data, title in (
        (ax1, uniform, "Uniform"),
        (ax2, normal, "Normal"),
        (ax3, exponential, "Exponential"),
    ):
        ax.hist(data, bins=30, edgecolor="black")
        ax.set_title(title)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.grid(True)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
