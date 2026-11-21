import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics import F1Score

from dl_lib.nn.module import Module
from dl_lib.vision.functional import load_cifar10_loaders
from dl_lib.vision.resnet import ResNet
from dl_lib.vision.vgg import Vgg


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
    model: torch.nn.Module | Module,
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


def plot(
    train_losses_plain: list,
    test_f1_micro_scores_plain: list,
    train_losses_resnet: list,
    test_f1_micro_scores_resnet: list,
    train_losses_vgg: list,
    test_f1_micro_scores_vgg: list,
):

    runs = {
        "plain CNN (depth=56)": (train_losses_plain, test_f1_micro_scores_plain),
        "ResNet (depth=56)": (train_losses_resnet, test_f1_micro_scores_resnet),
        "VGG (batch_norm=True)": (train_losses_vgg, test_f1_micro_scores_vgg),
    }

    fig, (ax_loss, ax_f1) = plt.subplots(1, 2, figsize=(12, 4))

    for label, (losses, f1s) in runs.items():
        ax_loss.plot(range(len(losses)), losses, marker="o", markersize=4, label=label)
        ax_f1.plot(range(len(f1s)), f1s, marker="o", markersize=4, label=label)

    ax_loss.set_title("Training loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("training loss")
    ax_loss.legend()

    ax_f1.set_title("Validation F1")
    ax_f1.set_xlabel("epoch")
    ax_f1.set_ylabel("validation F1")
    ax_f1.legend()

    fig.suptitle("The payoff: ResNet vs plain CNN vs VGG at matched depth")
    plt.tight_layout()
    plt.show()


def count_params(model: Module | torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> int:
    train_dataloader, val_dataloader, _, _ = load_cifar10_loaders(32)
    small_train = DataLoader(
        Subset(train_dataloader.dataset, range(1000)),
        batch_size=train_dataloader.batch_size,
        shuffle=True,
    )
    small_val_loader = DataLoader(
        Subset(val_dataloader.dataset, range(300)),
        batch_size=val_dataloader.batch_size,
        shuffle=True,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain = build_plain_cnn(depth=56).to(device)
    vgg = Vgg(batch_norm=True).to(device)
    resnet = ResNet(depth=56).to(device)
    _, _, test_f1_micro_scores_vgg, _, _, _, train_losses_vgg, _ = train_model(
        vgg,
        torch.optim.AdamW(vgg.parameters(), lr=0.0001),
        small_train,
        test_dataloader=small_val_loader,
        num_epochs=10,
        early_stopping_patience=20,
    )
    _, _, test_f1_micro_scores_plain, _, _, _, train_losses_plain, _ = train_model(
        plain,
        torch.optim.AdamW(plain.parameters(), lr=0.0001),
        small_train,
        test_dataloader=small_val_loader,
        num_epochs=10,
        early_stopping_patience=20,
    )
    _, _, test_f1_micro_scores_resnet, _, _, _, train_losses_resnet, _ = train_model(
        resnet,
        torch.optim.AdamW(resnet.parameters(), lr=0.0001),
        small_train,
        test_dataloader=small_val_loader,
        num_epochs=10,
        early_stopping_patience=20,
    )

    print(f"VGG (batch_norm=True): {count_params(vgg):,} params")
    print(
        f"  epoch=10 train_loss={train_losses_vgg[-1]} val_f1={test_f1_micro_scores_vgg[-1]}"
    )
    print(f"plain CNN (depth=56):: {count_params(plain):,} params")
    print(
        f"  epoch=10 train_loss={train_losses_plain[-1]} val_f1={test_f1_micro_scores_plain[-1]}"
    )
    print(f"ResNet (depth=56): {count_params(resnet):,} params")
    print(
        f"  epoch=10 train_loss={train_losses_resnet[-1]} val_f1={test_f1_micro_scores_resnet[-1]}"
    )
    print("Final val F1:")
    print(f"  VGG (batch_norm=True): {test_f1_micro_scores_vgg[-1]}")
    print(f"  plain CNN (depth=56): {test_f1_micro_scores_plain[-1]}")
    print(f"  ResNet (depth=56): {test_f1_micro_scores_resnet[-1]}")
    plot(
        train_losses_plain,
        test_f1_micro_scores_plain,
        train_losses_resnet,
        test_f1_micro_scores_resnet,
        train_losses_vgg,
        test_f1_micro_scores_vgg,
    )
    return 0
