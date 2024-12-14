import torch
import yaml
from thop import clever_format, profile
from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from src.datasets.dataset import get_cifar_dataset
from src.models.model import get_mlpmixer
from src.utils.logging import setup_wandb
from src.utils.lr_scheduler import get_scheduler
from src.utils.utils import set_seed


def setup_loss(cfg):
    """Set up the loss function based on the configuration.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        CrossEntropyLoss: The loss function.
    """
    return CrossEntropyLoss(label_smoothing=cfg["training"]["smooth_labels"])


def setup_optimizer(model, cfg):
    """Set up the optimizer based on the configuration.

    Args:
        model: The model to optimize.
        cfg (dict): Configuration dictionary.

    Returns:
        Adam: The optimizer.
    """
    return Adam(
        model.parameters(),
        lr=cfg["training"]["learning_rate"],
        weight_decay=cfg["training"]["weight_decay"],
    )


def setup_scheduler(optimizer, cfg):
    """Set up the learning rate scheduler based on the configuration.

    Args:
        optimizer: The optimizer for which to set up the scheduler.
        cfg (dict): Configuration dictionary.

    Returns:
        _LRScheduler: The learning rate scheduler.
    """
    return get_scheduler(optimizer, cfg)


def setup_device():
    """Return the device to use for computations.

    Returns:
        torch.device: The device to use for computations. If CUDA is available,
            returns a CUDA device, otherwise returns the CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_config(cfg_path):
    """Set up the configuration.

    Args:
        cfg_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed configuration dictionary.
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    print(f"Configuration loaded from {cfg_path}: ", cfg)
    return cfg


def setup_seed(cfg):
    """Set the seed for random number generation to ensure reproducibility.

    Args:
        cfg (dict): Configuration dictionary.
    """
    set_seed(cfg["training"]["seed"])


def setup_logging(cfg):
    """Set up the logging system.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        logging.Logger: The logger object.
    """
    return setup_wandb(cfg)


def setup_dataset(cfg):
    """Set up the dataset based on the configuration.

    Args:
        cfg (dict): Configuration dictionary.

    Returns:
        tuple: A tuple containing the training data loader and the validation data loader.
    """
    return get_cifar_dataset(cfg)


def setup_model(cfg):
    """Set up the model and calculate FLOPs and parameters."""
    model = get_mlpmixer(cfg)
    flops, params = get_flops(model, cfg)
    print(f"FLOPs: {flops}")
    print(f"Params: {params}")
    device = setup_device()
    return model.to(device)


def get_flops(model, cfg):
    """Compute the number of FLOPs and parameters in a model.

    Args:
        model: A PyTorch nn.Module object.

    Returns:
        tuple: A tuple containing the number of FLOPs and parameters in the model.
    """
    input = torch.randn(
        1, 3, cfg["training"]["image_size"], cfg["training"]["image_size"]
    )
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params
