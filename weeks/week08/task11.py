import csv

import torch
import torchvision
from torchmetrics import F1Score
from torchvision.datasets import FashionMNIST

from dl_lib.vision.alexnet import AlexNet
from dl_lib.vision.functional import load_fashion_mnist_loaders


def test_model(model, test_dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    f1_score_macro = F1Score(num_classes=10, average="macro", task="multiclass").to(
        device
    )
    f1_score_micro = F1Score(num_classes=10, average="micro", task="multiclass").to(
        device
    )
    total_f1_macro = 0.0
    total_f1_micro = 0.0
    total_loss = 0.0
    loss = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.max(outputs.data, 1)

            f1_macro = f1_score_macro(predicted.indices, labels)
            f1_micro = f1_score_micro(predicted.indices, labels)
            total_f1_macro += f1_macro.item()
            total_f1_micro += f1_micro.item()
            total_loss += loss(outputs, labels).item()
    return (
        total_f1_macro / len(test_dataloader),
        total_f1_micro / len(test_dataloader),
        total_loss / len(test_dataloader),
    )


def train_model(
    model: torch.nn.Module | AlexNet,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    early_stopping_patience: int = 3,
):
    assert isinstance(train_dataloader.dataset, FashionMNIST)
    assert isinstance(test_dataloader.dataset, FashionMNIST)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print("=================== Training Model ==================")
    print(f"Using device: {device}")
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
    patience_counter = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_dataloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_losses.append(running_loss / len(train_dataloader))
        test_f1_macro, test_f1_micro, test_loss = test_model(model, test_dataloader)
        test_losses.append(test_loss)
        test_f1_macro_scores.append(test_f1_macro)
        test_f1_micro_scores.append(test_f1_micro)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_test_f1_macro = test_f1_macro
            best_test_f1_micro = test_f1_micro
            best_train_loss = running_loss / len(train_dataloader)
            patience_counter = 0
        else:
            patience_counter += 1
        print(f"================{epoch + 1}/{num_epochs}================")
        print(f"Train Loss: {running_loss / len(train_dataloader)}")
        print(f"Test Loss: {test_loss}")
        print(f"Test F1 Macro: {test_f1_macro}")
        print(f"Test F1 Micro: {test_f1_micro}")
        if patience_counter >= early_stopping_patience:
            print(
                f"Early stopping triggered after {epoch + 1} epochs due to no improvement in test loss."
            )
            break

    return [
        best_train_loss,
        test_f1_macro_scores,
        test_f1_micro_scores,
        best_test_loss,
        best_test_f1_macro,
        best_test_f1_micro,
    ]


def save_to_csv(data: list[list], file_path: str) -> None:
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)


def main() -> int:
    train_dataloader, test_dataloader, class_names = load_fashion_mnist_loaders(32)
    model = AlexNet(in_channels=1, num_classes=len(class_names))
    columns = [
        "Model",
        "Train Loss",
        "Test F1 Macro",
        "Test F1 Micro",
        "Test Loss",
        "Best Test F1 Macro",
        "Best Test F1 Micro",
    ]
    res_my = ["My Model"] + train_model(
        model,
        torch.optim.AdamW(model.parameters(), lr=0.001),
        train_dataloader,
        test_dataloader,
        num_epochs=5,
        early_stopping_patience=3,
    )

    alexnet_model = torchvision.models.alexnet(num_classes=10)
    alexnet_model.features[0] = torch.nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=11,
        stride=4,
        padding=2,
    )
    res_torch = ["Torchvision AlexNet"] + train_model(
        alexnet_model,
        torch.optim.AdamW(alexnet_model.parameters(), lr=0.001),
        train_dataloader,
        test_dataloader,
        num_epochs=5,
        early_stopping_patience=3,
    )

    alexnet_smaller = torchvision.models.alexnet(
        weights=None,
        num_classes=10,
    )

    # Adapt first convolution for grayscale images
    alexnet_smaller.features[0] = torch.nn.Conv2d(
        1, 64, kernel_size=11, stride=4, padding=2
    )

    # Change classifier hidden dimensions
    alexnet_smaller.classifier[1] = torch.nn.Linear(256 * 6 * 6, 128)
    alexnet_smaller.classifier[4] = torch.nn.Linear(128, 128)
    alexnet_smaller.classifier[6] = torch.nn.Linear(128, 10)
    res_alexnet_smaller = ["Torchvision AlexNet Smaller"] + train_model(
        alexnet_smaller,
        torch.optim.AdamW(alexnet_smaller.parameters(), lr=0.001),
        train_dataloader,
        test_dataloader,
        num_epochs=5,
        early_stopping_patience=3,
    )

    save_to_csv(
        [columns, res_my, res_torch, res_alexnet_smaller], "results/week08/task11.csv"
    )
    return 0
