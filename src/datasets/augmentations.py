import random

import numpy as np
import torch


def cutmix(images, labels, alpha, prob):
    """Apply CutMix to a batch of data with a certain probability.

    Args:
        images: Tensor of images in the batch.
        labels: Tensor of labels in the batch.
        alpha: Hyperparameter for sampling the mix ratio from Beta distribution.
        prob: Probability of applying CutMix to an individual image.

    Returns:
        Augmented images and mixed labels.
    """
    batch_size = images.size(0)
    # Sample lambda from Beta distribution.
    lam = np.random.beta(alpha, alpha)
    indices = torch.randperm(batch_size)

    shuffled_images = images[indices]
    shuffled_labels = labels[indices]

    for i in range(batch_size):
        if random.random() < prob:
            _, H, W = images.size(1), images.size(2), images.size(3)
            cut_rat = np.sqrt(1.0 - lam)
            cut_w = int(W * cut_rat)
            cut_h = int(H * cut_rat)

            cx = np.random.randint(W)
            cy = np.random.randint(H)

            bbx1 = np.clip(cx - cut_w // 2, 0, W)
            bby1 = np.clip(cy - cut_h // 2, 0, H)
            bbx2 = np.clip(cx + cut_w // 2, 0, W)
            bby2 = np.clip(cy + cut_h // 2, 0, H)

            # Replace the region with the corresponding region from the shuffled images.
            images[i, :, bby1:bby2, bbx1:bbx2] = shuffled_images[
                i, :, bby1:bby2, bbx1:bbx2
            ]
            # Adjust labels to reflect the mix ratio for the modified image.
            labels[i] = lam * labels[i] + (1 - lam) * shuffled_labels[i]

    return images, labels


def mixup(images, labels, alpha, prob):
    """Apply MixUp to a batch of data with a certain probability.

    Args:
        images: Tensor of images in the batch.
        labels: Tensor of labels in the batch (one-hot encoded for correct mixing).
        alpha: Hyperparameter for sampling the mix ratio from Beta distribution.
        prob: Probability of applying MixUp to an individual image.

    Returns:
        Augmented images and mixed labels.
    """
    batch_size = images.size(0)
    new_images = images.clone()
    new_labels = labels.clone()

    for i in range(batch_size):
        if random.random() < prob:
            # Sample lambda from Beta distribution.
            lam = np.random.beta(alpha, alpha)
            index = torch.randint(0, batch_size, (1,)).item()

            # Mix current image with the randomly selected one.
            new_images[i] = lam * images[i] + (1 - lam) * images[index]
            new_labels[i] = lam * labels[i] + (1 - lam) * labels[index]

    return new_images, new_labels
