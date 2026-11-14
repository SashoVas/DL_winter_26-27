import torch

from dl_lib.vision.alexnet import AlexNet


def main() -> None:
    model = AlexNet()
    input = torch.randn(2, 3, 64, 64)

    model.train()
    first = model(input)
    second = model(input)
    print(f"Train mode: first logits {first.shape}, second logits {second.shape}")
    print(
        "Train mode: two forward passes identical?",
        torch.allclose(first, second, atol=1e-5),
    )

    model.eval()
    first = model(input)
    second = model(input)
    print(f"Eval mode: first logits {first.shape}, second logits {second.shape}")
    print(
        "Eval mode: two forward passes identical?",
        torch.allclose(first, second, atol=1e-5),
    )
