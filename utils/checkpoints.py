import os

import torch


def save_checkpoint(
    model, optimizer, epoch, checkpoint_dir, filename="best_ckpt.pth"
):
    """Saves a checkpoint of the model and optimizer state.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        epoch: Current epoch number.
        checkpoint_dir: Directory to save the checkpoint.
        filename: Name of the checkpoint file.
    """

    os.makedirs(checkpoint_dir, exist_ok=True)

    filepath = os.path.join(checkpoint_dir, filename)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        filepath,
    )


def load_checkpoint(model, filepath):
    """Loads a checkpoint of the model and optimizer state.

    Args:
        model: The model to load the checkpoint into.
        filepath: Path to the checkpoint file.
    Returns:
        model: The model with its state loaded from the checkpoint.
    """
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
