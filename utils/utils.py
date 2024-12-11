import os
import random

import numpy as np
import torch
import yaml
from thop import clever_format, profile


def set_seed(seed: int):
    """Set the seed for random number generation in Python, NumPy, and PyTorch.
    It also configures PyTorch to use deterministic algorithms and disables
    benchmarking for convolutional operations to ensure reproducibility.

    Args:
        seed (int): The seed value to ensure reproducibility.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    """Return the device to use for computations.

    Returns:
        torch.device: The device to use for computations. If CUDA is available,
            returns a CUDA device, otherwise returns the CPU device.
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_yaml_config(config_path):
    """Loads a configuration file in YAML format.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        cfg: Configuration dictionary loaded from the file.
    """
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_flops(model):
    """Compute the number of FLOPs and parameters in a model.

    Args:
        model: A PyTorch nn.Module object.
    Returns:
        tuple: A tuple containing the number of FLOPs and parameters in the model.
    """
    input = torch.randn(1, 3, 32, 32)
    flops, params = profile(model, inputs=(input,))
    flops, params = clever_format([flops, params], "%.3f")
    return flops, params
