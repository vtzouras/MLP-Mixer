import wandb


def setup_wandb(cfg):
    """Initializes the Weights & Biases logging system.

    Args:
        config: Configuration dictionary.

    Returns:
        wandb_logger: WandB logger object.
    """
    wandb.init(
        project=cfg["logging"]["wandb"]["project"],
        entity=cfg["logging"]["wandb"]["entity"],
        config=cfg,
    )
    return wandb


def log_metrics(
    logger, epoch, train_loss, train_accuracy, val_loss, val_accuracy
):
    """Logs metrics to WandB during training.

    Args:
        logger: The WandB logger object.
        epoch: Current epoch number.
        train_loss: Training loss for the current epoch.
        train_accuracy: Training accuracy for the current epoch.
        val_loss: Validation loss for the current epoch.
        val_accuracy: Validation accuracy for the current epoch.
    """
    logger.log(
        {
            "epoch": epoch,
            "train/loss": train_loss,
            "train/accuracy": train_accuracy,
            "val/loss": val_loss,
            "val/accuracy": val_accuracy,
        }
    )
