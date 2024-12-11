import math

from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        total_epochs,
        warmup_epochs,
        warmup_lr_start,
        base_lr,
        last_epoch=-1,
    ):
        """Custom Warmup and Cosine Annealing Learning Rate Scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            total_epochs (int): Total number of epochs for training.
            warmup_epochs (int): Number of epochs for linear warmup.
            warmup_lr_start (float): Starting learning rate during warmup.
            base_lr (float): Base learning rate after warmup.
            last_epoch (int, optional): The index of the last epoch. Default: -1.
        """
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.warmup_lr_start = warmup_lr_start
        self.base_lr = base_lr
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        """Compute learning rate using linear warmup and cosine annealing
        schedule.

        Returns:
            list: Learning rates for each parameter group.
        """
        current_epoch = (
            self.last_epoch + 1
        )  # Current epoch is 0-based index + 1.
        if current_epoch < self.warmup_epochs:
            # Warmup phase.
            return [
                (self.base_lr - self.warmup_lr_start)
                * current_epoch
                / self.warmup_epochs
                + self.warmup_lr_start
                for _ in self.base_lrs
            ]
        else:
            # Cosine annealing phase.
            return [
                base_lr
                * 0.5
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * (current_epoch - self.warmup_epochs)
                        / (self.total_epochs - self.warmup_epochs)
                    )
                )
                for base_lr in self.base_lrs
            ]
