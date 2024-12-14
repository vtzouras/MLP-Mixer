import argparse

from src.core.evaluator import inference
from src.core.setup import setup_config, setup_dataset, setup_model, setup_seed
from src.utils.utils import parse_yaml_config


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

    _, val_loader = setup_dataset(cfg)

    model = setup_model(cfg)

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
        class_names,
        plot=True,
        save_plot=True,
        save_dir="./plots",
    )


if __name__ == "__main__":
    main()
