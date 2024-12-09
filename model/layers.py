import torch
from einops.layers.torch import Rearrange
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 768,
    ):
        """Split the image into patches and embed them.

        Args:
            in_channels (int): Number of input channels.
            img_size (int): Size of the image (assumes square image).
            patch_size (int): Size of the patch (assumes square patch).
            embed_dim (int): Dimension of the embedding vector.
        Returns:
            torch.Tensor: Embedded patches.
        """
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size**2

        self.proj = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.embed_dim,
                kernel_size=self.patch_size,
                stride=self.patch_size,
            ),  # (batch, embed_dim, grid_H, grid_W)
            Rearrange(
                "b d (h) (w) -> b (h w) d"
            ),  # (batch, num_patches, embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class MLP(nn.Module):
    def __init__(
        self, embed_dim: int, hidden_features: int = 4, p: float = 0.0
    ):
        """Simple MLP with GELU activation.

        Args:
            in_features (int): Number of input features.
            hidden_features (int): Number of hidden features.
            out_features (int): Number of output features.
            p (float): Dropout probability.
        Returns:
            torch.Tensor: Output tensor.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_features = hidden_features
        self.out_features = embed_dim
        self.drop_p = p

        self.mlp = nn.Sequential(
            nn.Linear(self.embed_dim, self.hidden_features),
            nn.GELU("tanh"),
            nn.Dropout(self.drop_p),
            nn.Linear(self.hidden_features, self.out_features),
            nn.Dropout(self.drop_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class MixerLayer(nn.Module):
    def __init__(
        self, embed_dim, num_patches, token_dim, channel_dim, dropout_p
    ):
        """Mixer layer consists of token mixing and channel mixing. Token
        mixing MLP operates on patches (mix information across patches).
        Channel mixing MLP operates on per-patch embeddings (mix information
        across channels within each patch).

        Args:
            embed_dim (int): Dimension of the embedding vector.
            num_patches (int): Number of patches.
            token_dim (int): Dimension of the token mixing MLP.
            channel_dim (int): Dimension of the channel mixing MLP.
            dropout_p (float): Dropout probability.

        Returns:
            torch.Tensor: Output tensor.
        """
        super().__init__()
        self.token_mixing = nn.Sequential(
            nn.LayerNorm(embed_dim),
            Rearrange(
                "b p c -> b c p"
            ),  # (batch_size, num_patches, num_features) -> (batch_size, num_features, num_patches)
            MLP(num_patches, token_dim, dropout_p),
            Rearrange("b c p -> b p c"),
        )
        self.channel_mixing = nn.Sequential(
            nn.LayerNorm(embed_dim),
            MLP(
                embed_dim, channel_dim, dropout_p
            ),  # (batch_size, num_patches, num_features)
        )

    def forward(self, x):
        # Token mixing.
        x = x + self.token_mixing(x)
        # Channel mixing.
        x = x + self.channel_mixing(x)
        return x
