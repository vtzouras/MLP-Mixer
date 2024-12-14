from torch import nn

from src.models.layers import MixerLayer, PatchEmbedding


class MLPMixer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size,
        num_channels,
        embed_dim,
        num_classes,
        depth,
        token_dim,
        channel_dim,
        dropout_p,
    ):
        """MLPMixer model.

        Args:
            img_size (int): Size of the image (assumes square images).
            patch_size (int): Size of each patch (assumes square patches).
            num_channels (int): Number of input channels.
            embed_dim (int): Dimension of the embedding vector.
            num_classes (int): Number of classes.
            depth (int): Number of Mixer layers.
            token_dim (int): Dimension of the token mixing MLP.
            channel_dim (int): Dimension of the channel mixing MLP.
            dropout_p (float): Dropout probability.

        Returns:
            torch.Tensor: Output tensor.
        """
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            num_channels, img_size, patch_size, embed_dim
        )

        self.mixer_layers = nn.Sequential(
            *[
                MixerLayer(
                    embed_dim,
                    self.patch_embedding.num_patches,
                    token_dim,
                    channel_dim,
                    dropout_p,
                )
                for _ in range(depth)
            ]
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embedding(x)  # (batch_size, num_patches, embed_dim)
        x = self.mixer_layers(x)  # (batch_size, num_patches, embed_dim)
        x = self.layer_norm(x)  # (batch_size, num_patches, embed_dim)
        x = x.mean(dim=1)  # Global average pooling over patches.
        x = self.classifier(x)  # (batch_size, num_classes)
        return x


def get_mlpmixer(cfg):
    """Initialize a Vision Transformer based on configuration.

    Args:
        cfg: Configuration dictionary.

    Returns:
        nn.Module: Vision Transformer model.
    """
    return MLPMixer(
        img_size=cfg["model"]["image_size"],
        patch_size=cfg["model"]["patch_size"],
        num_channels=cfg["model"]["num_channels"],
        embed_dim=cfg["model"]["embed_dim"],
        num_classes=cfg["model"]["num_classes"],
        depth=cfg["model"]["depth"],
        token_dim=cfg["model"]["token_dim"],
        channel_dim=cfg["model"]["channel_dim"],
        dropout_p=cfg["model"]["dropout"],
    )
