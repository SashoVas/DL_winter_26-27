import matplotlib.pyplot as plt
import numpy as np

from dl_lib.metrics import functional


def main() -> None:
    true = np.array([0] * 80 + [1] * 20)
    probs = np.concatenate(
        [np.random.uniform(0.0, 0.8, 80), np.random.uniform(0.2, 1, 20)]
    )

    thresholds = np.linspace(0.05, 0.95, 19)
    precisions = []
    recalls = []
    for threshold in thresholds:
        pred = functional.apply_threshold(probs, threshold)
        precisions.append(functional.precision(true, pred))
        recalls.append(functional.recall(true, pred))

    for i in [0, 9, 18]:
        print(
            f"threshold={thresholds[i]}: precision={precisions[i]}, recall={recalls[i]}"
        )

    plt.plot(thresholds, precisions, marker="o", label="Precision")
    plt.plot(thresholds, recalls, marker="o", label="Recall")
    plt.xlabel("threshold")
    plt.ylabel("value")
    plt.title("Precision and recal as the classification threshold rises")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()
