import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
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


def conv_bn_relu(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
):
    return [
        torch.nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        ),
        torch.nn.BatchNorm2d(out_channels),
        torch.nn.ReLU(),
    ]


def build_plain_cnn(depth: int) -> torch.nn.Module:
    res = conv_bn_relu(3, 16)
    in_ch = 16
    for stage, out_ch in enumerate([16, 32, 64]):
        for i in range((depth - 2) // 6):
            stride = 2 if (stage > 0 and i == 0) else 1
            res += conv_bn_relu(in_ch, out_ch, stride=stride)
            in_ch = out_ch

    res += [
        torch.nn.AdaptiveAvgPool2d((1, 1)),
        torch.nn.Flatten(),
        torch.nn.Linear(64, 10),
    ]
    return torch.nn.Sequential(*res)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    train_dataloader, _, _, class_names = load_cifar10_loaders(32)
    small_loader = DataLoader(
        Subset(train_dataloader.dataset, range(100)),
        batch_size=train_dataloader.batch_size,
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    shallow_cnn = build_plain_cnn(20).to(device)
    deep_cnn = build_plain_cnn(56).to(device)
    _, _, _, _, _, _, train_losses_shallow, _ = train_model(
        shallow_cnn,
        torch.optim.Adam(shallow_cnn.parameters(), lr=0.00005),
        small_loader,
        small_loader,
        num_epochs=5,
        early_stopping_patience=20,
        num_classes=len(class_names),
    )

    _, _, _, _, _, _, train_losses_deep, _ = train_model(
        shallow_cnn,
        torch.optim.Adam(shallow_cnn.parameters(), lr=0.00005),
        small_loader,
        small_loader,
        num_epochs=5,
        early_stopping_patience=20,
        num_classes=len(class_names),
    )
    print(f"Plain CNN depth=20: {count_params(shallow_cnn):,} params")
    print(f"  depth=20 epoch=5 train_loss={train_losses_shallow[-1]}")
    print(f"Plain CNN depth=56: {count_params(deep_cnn):,} params")
    print(f"  depth=56 epoch=5 train_loss={train_losses_deep[-1]}")
    plt.figure(figsize=(6, 4))
    plt.plot(
        range(1, len(train_losses_shallow) + 1),
        train_losses_shallow,
        marker="o",
        label="plain CNN depth=20",
    )
    plt.plot(
        range(1, len(train_losses_deep) + 1),
        train_losses_deep,
        marker="o",
        label="plain CNN depth=56",
    )
    plt.xlabel("epoch")
    plt.ylabel("training loss")
    plt.title("The degradation problem: deeper plain CNN trains worse")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return 0
