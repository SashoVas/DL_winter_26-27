import numpy as np
import torch

from dl_lib.activations.functional import softmax


def main() -> None:
    scores = np.array([1.0, 2.0, 3.0])
    sm = softmax(scores)
    torch_sm = torch.softmax(torch.from_numpy(scores), dim=0)
    if not np.allclose(sm, torch_sm):
        raise ValueError("Softmax values are different")
    print("scores:", scores)
    print("softmax:", sm)
    print("sum of probabilities:", sm.sum())
    print()
    scores = np.array([1000.0, 1001.0, 1002.0])
    print("Large scores:", scores)
    print("Naive softmax (unstable):", np.exp(scores) / np.exp(scores).sum())
    print("Stable softmax:", softmax(scores))
