import copy
import csv

import torch
import torchvision
from torch.utils.data import Subset
from torchmetrics import F1Score

from dl_lib.vision.alexnet import AlexNet
from dl_lib.vision.functional import load_clouds_loaders


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
    model: torch.nn.Module | AlexNet,
    optimizer: torch.optim.Optimizer,
    train_dataloader: torch.utils.data.DataLoader,
    test_dataloader: torch.utils.data.DataLoader,
    num_epochs: int = 5,
    early_stopping_patience: int = 3,
    num_classes: int = 10,
):
    assert isinstance(train_dataloader.dataset, torchvision.datasets.ImageFolder)
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
            if hasattr(model, "state_dict"):
                best_model_state = copy.deepcopy(model.state_dict())

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
    ]


def save_to_csv(data: list[list], file_path: str) -> None:
    with open(file_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerows(data)


class myModel(torch.nn.Module):
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=3,
                stride=2,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.Conv2d(
                in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=0),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=0
            ),
        )
        self.classifier_first_part = torch.nn.Sequential(
            torch.nn.Linear(256 * 1 * 1, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.classifier_second_part = torch.nn.Sequential(
            torch.nn.Linear(256, 256),
            torch.nn.LayerNorm(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.1),
        )
        self.classifier_last_part = torch.nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier_first_part(x) + x
        x = self.classifier_second_part(x) + x
        x = self.classifier_last_part(x)
        return x


def main() -> int:
    in_channels = 3
    epochs = 20
    train_dataloader, val_dataloader, test_dataloader, class_names = (
        load_clouds_loaders(32)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AlexNet(in_channels=in_channels, num_classes=len(class_names)).to(device)
    columns = [
        "Model",
        "Train Loss",
        "Validation F1 Macro",
        "Validation F1 Micro",
        "Validation Loss",
        "Best Validation F1 Macro",
        "Best Validation F1 Micro",
    ]
    res_my = ["AlexNet My implementation"] + train_model(
        model,
        torch.optim.Adam(model.parameters(), lr=0.00005),
        train_dataloader,
        val_dataloader,
        num_epochs=epochs,
        early_stopping_patience=2,
        num_classes=len(class_names),
    )
    alexnet_model = torchvision.models.alexnet(num_classes=len(class_names))
    alexnet_model.features[0] = torch.nn.Conv2d(
        in_channels=in_channels,
        out_channels=64,
        kernel_size=11,
        stride=4,
        padding=2,
    )
    alexnet_model = alexnet_model.to(device)
    res_torch = ["Torchvision AlexNet"] + train_model(
        alexnet_model,
        torch.optim.Adam(alexnet_model.parameters(), lr=0.00005),
        train_dataloader,
        val_dataloader,
        num_epochs=epochs,
        early_stopping_patience=2,
        num_classes=len(class_names),
    )
    alexnet_smaller = torchvision.models.alexnet(
        weights=None,
        num_classes=len(class_names),
    )
    alexnet_smaller.features[0] = torch.nn.Conv2d(
        in_channels=in_channels, out_channels=64, kernel_size=11, stride=4, padding=2
    )
    alexnet_smaller.classifier[1] = torch.nn.Linear(256 * 6 * 6, 128)
    alexnet_smaller.classifier[4] = torch.nn.Linear(128, 128)
    alexnet_smaller.classifier[6] = torch.nn.Linear(128, len(class_names))
    alexnet_smaller.to(device)
    res_alexnet_smaller = ["Torchvision AlexNet Smaller"] + train_model(
        alexnet_smaller,
        torch.optim.Adam(alexnet_smaller.parameters(), lr=0.00005),
        train_dataloader,
        val_dataloader,
        num_epochs=epochs,
        early_stopping_patience=2,
        num_classes=len(class_names),
    )

    my_model = myModel(in_channels=in_channels, num_classes=len(class_names)).to(device)
    res_my_model = ["My Model"] + train_model(
        my_model,
        torch.optim.Adam(my_model.parameters(), lr=0.00005),
        train_dataloader,
        val_dataloader,
        num_epochs=epochs,
        early_stopping_patience=2,
        num_classes=len(class_names),
    )
    save_to_csv(
        [columns, res_my, res_torch, res_alexnet_smaller, res_my_model],
        "results/week08/task15.csv",
    )

    print(
        "Best model on test set:",
        test_model(my_model, test_dataloader, num_classes=len(class_names)),
    )
    return 0
