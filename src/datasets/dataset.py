import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import RandAugment, RandomErasing

from src.datasets.augmentations import cutmix, mixup


def get_cifar_dataset(cfg):
    """Returns CIFAR-10 train and validation dataloaders with CutMix, MixUp,
    RandAugment, and Random Erasing.

    Args:
        cfg: Configuration dictionary containing batch size, data paths, and augmentation options.

    Returns:
        train_loader, val_loader: DataLoader objects for train and validation datasets.
    """
    preprocess_transform = [transforms.Resize((32, 32))]
    # RandAugment requires a PIL image.
    if cfg["augmentation"]["randaugment"]:
        preprocess_transform.append(
            RandAugment(
                num_ops=cfg["augmentation"]["randaugment_num_ops"],
                magnitude=cfg["augmentation"]["randaugment_magnitude"],
            )
        )
    preprocess_transform = transforms.Compose(preprocess_transform)

    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]
            ),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.491, 0.482, 0.446], std=[0.247, 0.243, 0.261]
            ),
        ]
    )

    if cfg["augmentation"]["random_erasing"]:
        random_erasing = RandomErasing(
            p=cfg["augmentation"]["random_erasing_prob"],
            scale=cfg["augmentation"]["random_erasing_scale"],
            ratio=cfg["augmentation"]["random_erasing_ratio"],
            value=cfg["augmentation"]["random_erasing_value"],
        )
    else:
        random_erasing = None

    def train_transform(img):
        """Applies the preprocessing and base transforms to an image.

        Args:
            img: PIL image to be transformed.

        Returns:
            Transformed image as a torch tensor.
        """
        img = preprocess_transform(img)
        img = base_transform(img)
        return img

    def collate_fn(batch):
        """Collate function for the dataloader. Applies CutMix or MixUp if
        specified in the config. Applies Random Erasing if specified in the
        config.

        Args:
            batch: List of tuples containing the image and its label.

        Returns:
            images: Transformed images as a torch tensor.
            labels: Labels as a torch tensor.
        """
        images, labels = zip(*batch)
        images = torch.stack([train_transform(img) for img in images])
        labels = torch.tensor(labels)

        # Apply CutMix or MixUp.
        if cfg["augmentation"]["cutmix"]:
            images, labels = cutmix(
                images,
                labels,
                alpha=cfg["augmentation"]["mix_alpha"],
                prob=cfg["augmentation"]["cutmix_prob"],
            )

        elif cfg["augmentation"]["mixup"]:
            images, labels = mixup(
                images,
                labels,
                alpha=cfg["augmentation"]["mix_alpha"],
                prob=cfg["augmentation"]["mixup_prob"],
            )

        if random_erasing:
            images = torch.stack([random_erasing(img) for img in images])

        return images, labels

    train_dataset = datasets.CIFAR10(
        root=cfg["data"]["train_dir"], train=True, download=True
    )
    val_dataset = datasets.CIFAR10(
        root=cfg["data"]["test_dir"],
        train=False,
        download=True,
        transform=val_transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader
