from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from src.utils.warmupcosinelr import WarmupCosineLR


def get_scheduler(optimizer, cfg):
    """Get learning rate scheduler based on configuration.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        cfg (dict): Configuration dictionary.

    Returns:
        _LRScheduler: Learning rate scheduler.
    """
    scheduler_type = cfg["training"]["scheduler"]
    if scheduler_type == "WarmupCosineLR":
        return WarmupCosineLR(
            optimizer,
            total_epochs=cfg["training"]["epochs"],
            warmup_epochs=cfg["training"]["warmup_epochs"],
            warmup_lr_start=cfg["training"]["warmup_lr_start"],
            base_lr=cfg["training"]["learning_rate"],
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])
    elif scheduler_type == "step":
        return StepLR(
            optimizer,
            step_size=cfg["training"].get("step_size", 10),
            gamma=cfg["training"].get("gamma", 0.1),
        )
    elif scheduler_type is None:
        return None
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")
