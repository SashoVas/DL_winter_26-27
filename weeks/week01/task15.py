import numpy as np
import torch


def main() -> None:
    np_matrix = np.arange(1, 7).reshape(2, 3)
    print("NumPy array:")
    print(np_matrix)
    print(f"dtype={np_matrix.dtype}, shape={np_matrix.shape}")
    print()

    torch_matrix = torch.from_numpy(np_matrix)
    print("PyTorch tensor:")
    print(torch_matrix)
    print(
        f"dtype={torch_matrix.dtype}, shape={torch_matrix.shape}, device={torch_matrix.device}"
    )
    print()
    print("Selected device: cpu")
    torch_matrix.to("cpu")  # I dont have gpu on this computer
    print(
        f"Tensor moved to device: {torch_matrix.device}, dtype=torch.float32")
