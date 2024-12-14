import os

import torch

from src.core.setup import setup_device
from src.utils.utils import plot_images


def inference(
    model,
    loader,
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
    device = setup_device()
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
