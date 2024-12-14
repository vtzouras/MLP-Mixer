import torch
from tqdm import tqdm

from src.core.setup import setup_device
from src.utils.checkpoints import save_checkpoint
from src.utils.logging import log_metrics


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler=None,
        logger=None,
        cfg=None,
    ):
        """Initialize the Trainer class.

        Args:
            model: The model to be trained.
            train_loader: DataLoader for the training dataset.
            val_loader: DataLoader for the validation dataset.
            optimizer: Optimizer for training the model.
            criterion: Loss function used during training.
            scheduler (optional): Learning rate scheduler.
            logger (optional): Logger for tracking metrics during training.
            cfg (optional): Configuration dictionary.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger
        self.cfg = cfg
        self.device = setup_device()

    def train_one_epoch(self, epoch):
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the average loss and accuracy for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        progress_bar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc=f"Epoch {epoch}",
        )
        for _, (images, labels) in progress_bar:
            images, labels = images.to(self.device), labels.to(self.device)

            # Forward pass.
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            # Backward pass.
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate loss and correct predictions.
            total_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

            progress_bar.set_postfix(
                {
                    "train_loss": loss.item(),
                    "accuracy": correct_preds / total_samples,
                }
            )

        avg_loss = total_loss / total_samples
        accuracy = correct_preds / total_samples
        return avg_loss, accuracy

    def validate(self):
        """Evaluate the model on the validation set.

        Returns:
            avg_loss (float): The average loss on the validation set.
            accuracy (float): The accuracy on the validation set.
        """
        self.model.eval()
        total_loss = 0.0
        correct_preds = 0
        total_samples = 0

        with torch.no_grad():
            progress_bar = tqdm(
                self.val_loader, total=len(self.val_loader), desc="Validation"
            )
            for images, labels in progress_bar:
                images, labels = images.to(self.device), labels.to(self.device)

                # Forward pass.
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

                # Accumulate loss and correct predictions.
                total_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                correct_preds += (preds == labels).sum().item()
                total_samples += labels.size(0)

        # Average loss and accuracy.
        avg_loss = total_loss / total_samples
        accuracy = correct_preds / total_samples
        return avg_loss, accuracy

    def train(self, epochs):
        """Train the model for the specified number of epochs.

        Args:
            epochs (int): Number of epochs to train for.
        """
        best_val_accuracy = 0.0

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_one_epoch(epoch)
            val_loss, val_accuracy = self.validate()

            if self.logger:
                log_metrics(
                    self.logger,
                    epoch,
                    train_loss,
                    train_accuracy,
                    val_loss,
                    val_accuracy,
                )

            # Save checkpoint if accuracy improves.
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    self.cfg["data"]["checkpoint_dir"],
                )

            # Step LR scheduler (handle ReduceLROnPlateau).
            if isinstance(
                self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.scheduler.step(val_loss)
            elif self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch {epoch+1}/{epochs}, "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}"
            )
