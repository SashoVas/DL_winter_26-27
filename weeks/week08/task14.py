import matplotlib.pyplot as plt

from dl_lib.vision.functional import load_clouds_loaders


def main() -> int:
    train_dataloader, _, _, class_names = load_clouds_loaders(8)

    images, labels = next(iter(train_dataloader))

    _, axes = plt.subplots(1, len(images), figsize=(16, 3))

    for image, label, ax in zip(images, labels, axes):
        image = image.permute(1, 2, 0).numpy()

        ax.imshow(image)
        ax.set_title(class_names[label.item()])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

    return 0
