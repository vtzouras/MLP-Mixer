import argparse

from src.core.setup import (
    setup_config,
    setup_dataset,
    setup_logging,
    setup_loss,
    setup_model,
    setup_optimizer,
    setup_scheduler,
    setup_seed,
)
from src.core.trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to the config file",
    )
    args = parser.parse_args()

    # Load configuration.
    cfg = setup_config(args.config)

    # Set up random seed.
    setup_seed(cfg)

    # Set up logging.
    logger = setup_logging(cfg)

    # Set up dataset and model.
    train_loader, val_loader = setup_dataset(cfg)
    model = setup_model(cfg)

    # Set up loss, optimizer, and scheduler.
    criterion = setup_loss(cfg)
    optimizer = setup_optimizer(model, cfg)
    scheduler = setup_scheduler(optimizer, cfg)

    # Initialize and run the trainer.
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        logger=logger,
        cfg=cfg,
    )
    trainer.train(cfg["training"]["epochs"])


if __name__ == "__main__":
    main()
