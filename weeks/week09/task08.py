from itertools import pairwise

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.flop_counter import FlopCounterMode
from torchmetrics import F1Score

from dl_lib.vision.functional import load_cifar10_loaders


def test_model(model, test_dataloader, num_classes=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    f1_score_macro = F1Score(
        num_classes=num_classes, average="macro", task="multiclass"
    ).to(device)
    f1_score_micro = F1Score(
        num_classes=num_classes, average="micro", task="multiclass"
    ).to(device)

    total_loss = 0.0
    total_samples = 0
    loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            f1_score_macro.update(predicted, labels)
            f1_score_micro.update(predicted, labels)
            batch_size = labels.size(0)
            total_loss += loss(outputs, labels).item() * batch_size
            total_samples += batch_size
    return (
        f1_score_macro.compute().item(),
        f1_score_micro.compute().item(),
        total_loss / total_samples,
    )


def train_model(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    early_stopping_patience: int = 3,
    num_classes: int = 10,
):
    assert isinstance(train_dataloader.dataset, Subset)
    assert isinstance(test_dataloader.dataset, Subset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=================== Training Model ==================")
    print(f"Train size: {len(train_dataloader.dataset)}")
    print(f"Test size: {len(test_dataloader.dataset)}")
    print()
    criterion = torch.nn.CrossEntropyLoss()
    train_losses = []
    test_losses = []
    test_f1_macro_scores = []
    test_f1_micro_scores = []
    best_test_loss = float("inf")
    best_test_f1_macro = 0.0
    best_test_f1_micro = 0.0
    best_train_loss = float("inf")
    best_model_state = None
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        total_samples = 0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total_samples += batch_size

        train_losses.append(running_loss / total_samples)
        test_f1_macro, test_f1_micro, test_loss = test_model(
            model, test_dataloader, num_classes
        )
        test_losses.append(test_loss)
        test_f1_macro_scores.append(test_f1_macro)
        test_f1_micro_scores.append(test_f1_micro)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_f1_macro = test_f1_macro
            best_test_f1_micro = test_f1_micro
            best_train_loss = running_loss / total_samples

            patience_counter = 0
        else:
            patience_counter += 1
        print(f"================{epoch + 1}/{num_epochs}================")
        print(f"Train Loss: {running_loss / total_samples}")
        print(f"Test Loss: {test_loss}")
        print(f"Test F1 Macro: {test_f1_macro}")
        print(f"Test F1 Micro: {test_f1_micro}")
        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs due to no improvement in test loss."
            )
            break
    if best_model_state is not None and hasattr(model, "load_state_dict"):
        model.load_state_dict(best_model_state)
    return [
        best_train_loss,
        test_f1_macro_scores,
        test_f1_micro_scores,
        best_test_loss,
        best_test_f1_macro,
        best_test_f1_micro,
        train_losses,
        test_losses,
    ]


CHANNELS = [16, 32, 64, 128]


def make_cnn(strided: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(3, CHANNELS[0], 3, padding=1),
        nn.BatchNorm2d(CHANNELS[0]),
        nn.ReLU(),
    ]
    for in_channels, out_channels in pairwise(CHANNELS):
        if strided:
            layers += [
                nn.Conv2d(in_channels, in_channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(),
            ]
        else:
            layers.append(nn.MaxPool2d(2))
        layers += [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(CHANNELS[-1], 10)]
    return nn.Sequential(*layers)


def conv_flops(model: nn.Module) -> int:
    counter = FlopCounterMode(display=False)
    model.eval()
    with counter, torch.no_grad():
        model(torch.randn(1, 3, 64, 64))
    return counter.get_flop_counts()["Global"][torch.ops.aten.convolution]


def main():
    train_dataloader, val_dataloader, _, _ = load_cifar10_loaders(32)
    small_train = DataLoader(
        Subset(train_dataloader.dataset, range(1000)),
        batch_size=train_dataloader.batch_size,
        shuffle=True,
    )
    small_val = DataLoader(
        Subset(val_dataloader.dataset, range(1000)),
        batch_size=val_dataloader.batch_size,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(0)
    model = make_cnn(False)
    params = sum(p.numel() for p in model.parameters())
    flops = conv_flops(model)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    results = train_model(model, optimizer, small_train, small_val, num_epochs=10)
    final_f1 = results[1][-1]
    final_loss = results[-2][-1]

    torch.manual_seed(0)
    model_strided = make_cnn(True)
    params_strided = sum(p.numel() for p in model_strided.parameters())
    flops_strided = conv_flops(model_strided)
    model_strided.to(device)
    optimizer = torch.optim.Adam(model_strided.parameters(), lr=1e-3)
    results = train_model(
        model_strided, optimizer, small_train, small_val, num_epochs=10
    )
    final_f1_strided = results[1][-1]
    final_loss_strided = results[-2][-1]
    print(f"MaxPool downsampler:  {params:,} params, {flops:,} conv FLOPs per image")
    print(
        f"Stride-2 downsampler::  {params_strided:,} params, {flops_strided:,} conv FLOPs per image"
    )
    print(
        f"Stride-2 uses {params_strided / params}x the params and {flops_strided / flops}x the FLOPs of MaxPool."
    )
    print(f"MaxPool downsampler: final train_loss={final_loss} val_f1={final_f1}")
    print(
        f"Stride-2 downsampler: final train_loss={final_loss_strided} val_f1={final_f1_strided}"
    )
