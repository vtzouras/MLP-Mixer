import os

import matplotlib.pyplot as plt
import torch

from dataset.dataset import get_cifar_dataset
from model.model import get_mlpmixer
from utils.checkpoints import load_checkpoint
from utils.utils import get_device, parse_yaml_config


def plot_images(
    images, labels, preds, class_names, num_images=10, save_path=None
):
    """Plot a grid of images from the given tensor, along with their
    corresponding labels and predictions.

    Args:
        images (torch.Tensor): A tensor of shape (num_images, channels, height, width) containing the images to plot.
        labels (torch.Tensor): A tensor of shape (num_images,) containing the true labels for the images.
        preds (torch.Tensor): A tensor of shape (num_images,) containing the predicted labels for the images.
        class_names (list[str]): A list of strings containing the names of the classes.
        num_images (int, optional): The number of images to plot. Defaults to 10.
        save_path (str, optional): The path to save the plot. Defaults to None, which means the plot is not saved but displayed instead.
    """
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    for i in range(num_images):
        ax = axes[i]
        image = (
            images[i].cpu().numpy().transpose(1, 2, 0)
        )  # Convert to HWC format.
        ax.imshow(image)
        true_label = class_names[labels[i].item()]
        pred_label = class_names[preds[i].item()]
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


# 86.72% accuracy
def inference(
    model,
    loader,
    device,
    class_names,
    plot=False,
    save_plot=False,
    save_dir="./plots",
):
    """Evaluate the model on a given dataset.

    Args:
        model: The model to evaluate.
        loader: The DataLoader for the evaluation dataset.
        device: The device to run the evaluation on.
        class_names: A list of strings containing the names of the classes.
        plot (bool, optional): Whether to plot a few images with their
            corresponding labels and predictions. Defaults to False.
        save_plot (bool, optional): Whether to save the plot to a file.
            Defaults to False.
        save_dir (str, optional): The directory to save the plot to.
            Defaults to "./plots".
    Returns:
        accuracy: The accuracy of the model on the evaluation dataset.
    """
    model.eval()
    correct_preds = 0
    total_samples = 0

    images_list = []
    labels_list = []
    preds_list = []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

            images_list.append(images)
            labels_list.append(labels)
            preds_list.append(preds)

            if plot and len(images_list) >= 1:
                break

    accuracy = correct_preds / total_samples
    print(f"Inference Accuracy: {accuracy * 100:.2f}%")

    if plot:
        images = torch.cat(images_list)[:10]
        labels = torch.cat(labels_list)[:10]
        preds = torch.cat(preds_list)[:10]

        if save_plot:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, "inference_plot.png")
            plot_images(
                images, labels, preds, class_names, save_path=save_path
            )
        else:
            plot_images(images, labels, preds, class_names)

    return accuracy


def main():
    config_path = "configs/default.yaml"
    cfg = parse_yaml_config(config_path)

    model = get_mlpmixer(cfg)
    model = load_checkpoint(model, "checkpoints/best_ckpt.pth")
    device = get_device()
    model.to(device)

    _, val_loader = get_cifar_dataset(cfg)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    inference(
        model,
        val_loader,
        device,
        class_names,
        plot=True,
        save_plot=True,
        save_dir="./plots",
    )


if __name__ == "__main__":
    main()
