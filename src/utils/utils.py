import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    """Set the seed for random number generation in Python, NumPy, and PyTorch.
    It also configures PyTorch to use deterministic algorithms and disables
    benchmarking for convolutional operations to ensure reproducibility.

    Args:
        seed (int): The seed value to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_yaml_config(config_path):
    """Loads a configuration file in YAML format.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        cfg: Configuration dictionary loaded from the file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


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
