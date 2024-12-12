from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from dataset.dataset import get_cifar_dataset
from model.model import get_mlpmixer
from trainer.trainer import Trainer
from utils.logging import setup_wandb
from utils.lr_scheduler import get_scheduler
from utils.utils import get_device, get_flops, parse_yaml_config, set_seed


def main():
    config_path = "configs/default.yaml"
    cfg = parse_yaml_config(config_path)
    print("Config loaded: ", cfg)

    device = get_device()

    # Set random seed for reproducibility.
    set_seed(cfg["training"]["seed"])

    # Set up logging.
    logger = setup_wandb(cfg)

    # Load datasets.
    train_loader, val_loader = get_cifar_dataset(cfg)

    # Initialize model, loss, optimizer, and scheduler.
    model = get_mlpmixer(cfg)
    flops, params = get_flops(model)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    model.to(device)
    criterion = CrossEntropyLoss(
        label_smoothing=cfg["training"]["smooth_labels"]
    )
    optimizer = Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    scheduler = get_scheduler(optimizer, cfg)

    # Initialize trainer.
    trainer = Trainer(
        model,
        train_loader,
        val_loader,
        optimizer,
        criterion,
        scheduler,
        logger,
        cfg,
    )

    # Start training.
    trainer.train(cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
