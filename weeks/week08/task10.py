import matplotlib.pyplot as plt

from dl_lib.vision.functional import load_fashion_mnist_loaders


def main() -> int:
    train_dataloader, _, class_names = load_fashion_mnist_loaders(8)

    images, labels = next(iter(train_dataloader))

    _, axes = plt.subplots(1, len(images), figsize=(16, 3))

    for image, label, ax in zip(images, labels, axes):
        ax.imshow(image.squeeze().cpu().numpy(), cmap="gray")
        ax.set_title(class_names[label.item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return 0
