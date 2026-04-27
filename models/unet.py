"""
UNet architecture for diffusion models.

Based on ShuffleNet v2 blocks (Ma et al., 2018) for lightweight
encoder/decoder with channel-split, depthwise convolution, and
channel-shuffle operations.

Reference: https://arxiv.org/pdf/1807.11164.pdf
"""

import torch
import torch.nn as nn


class ConvBnSiLu(nn.Module):
    """Conv2d → BatchNorm → SiLU activation."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x):
        return self.module(x)


class ChannelShuffle(nn.Module):
    """Shuffle channels between groups for cross-group information flow."""

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.groups, c // self.groups, h, w)
        x = x.transpose(1, 2).contiguous().view(n, -1, h, w)
        return x


class ResidualBottleneck(nn.Module):
    """ShuffleNet v2 basic unit: split → two branches → concat → shuffle."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels // 2, in_channels // 2, 1, 1, 0),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, 1, 1, groups=in_channels // 2),
            nn.BatchNorm2d(in_channels // 2),
            ConvBnSiLu(in_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1), self.branch2(x2)], dim=1)
        return self.channel_shuffle(x)


class ResidualDownsample(nn.Module):
    """ShuffleNet v2 spatial downsampling unit (stride 2)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 2, 1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
        )
        self.branch2 = nn.Sequential(
            ConvBnSiLu(in_channels, out_channels // 2, 1, 1, 0),
            nn.Conv2d(out_channels // 2, out_channels // 2, 3, 2, 1, groups=out_channels // 2),
            nn.BatchNorm2d(out_channels // 2),
            ConvBnSiLu(out_channels // 2, out_channels // 2, 1, 1, 0),
        )
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        return self.channel_shuffle(x)


class TimeMLP(nn.Module):
    """Inject timestep information into feature maps via MLP + residual add."""

    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x, t):
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        return self.act(x + t_emb)


class EncoderBlock(nn.Module):
    """Encoder stage: residual bottlenecks → time conditioning → downsample."""

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for _ in range(3)],
            ResidualBottleneck(in_channels, out_channels // 2),
        )
        self.time_mlp = TimeMLP(time_embedding_dim, out_channels, out_channels // 2)
        self.conv1 = ResidualDownsample(out_channels // 2, out_channels)

    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x_shortcut = self.time_mlp(x_shortcut, t)
        x = self.conv1(x_shortcut)
        return x, x_shortcut


class DecoderBlock(nn.Module):
    """Decoder stage: upsample → skip connection → residual bottlenecks → time conditioning."""

    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv0 = nn.Sequential(
            *[ResidualBottleneck(in_channels, in_channels) for _ in range(3)],
            ResidualBottleneck(in_channels, in_channels // 2),
        )
        self.time_mlp = TimeMLP(time_embedding_dim, in_channels, in_channels // 2)
        self.conv1 = ResidualBottleneck(in_channels // 2, out_channels // 2)

    def forward(self, x, x_shortcut, t=None):
        x = self.upsample(x)
        x = torch.cat([x, x_shortcut], dim=1)
        x = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x, t)
        return self.conv1(x)


class Unet(nn.Module):
    """
    UNet with ShuffleNet v2 blocks and sinusoidal time embeddings.

    Architecture:
        init_conv → [EncoderBlock × N] → mid_block → [DecoderBlock × N] → final_conv

    Args:
        timesteps: Maximum number of diffusion timesteps.
        time_embedding_dim: Dimension of the learned time embedding.
        in_channels: Input image channels.
        out_channels: Output image channels (typically same as in_channels).
        base_dim: Base channel dimension (must be even).
        dim_mults: List of channel multipliers for each encoder/decoder stage.
    """

    def __init__(
        self,
        timesteps,
        time_embedding_dim,
        in_channels=3,
        out_channels=2,
        base_dim=32,
        dim_mults=(2, 4, 8, 16),
    ):
        super().__init__()
        assert isinstance(dim_mults, (list, tuple))
        assert base_dim % 2 == 0, "base_dim must be even."

        channels = self._compute_channel_pairs(base_dim, dim_mults)

        self.init_conv = ConvBnSiLu(in_channels, base_dim, 3, 1, 1)
        self.time_embedding = nn.Embedding(timesteps, time_embedding_dim)

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(c_in, c_out, time_embedding_dim) for c_in, c_out in channels
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(c_out, c_in, time_embedding_dim) for c_in, c_out in reversed(channels)
        ])
        self.mid_block = nn.Sequential(
            *[ResidualBottleneck(channels[-1][1], channels[-1][1]) for _ in range(2)],
            ResidualBottleneck(channels[-1][1], channels[-1][1] // 2),
        )
        self.final_conv = nn.Conv2d(channels[0][0] // 2, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        x = self.init_conv(x)
        if t is not None:
            t = self.time_embedding(t)

        shortcuts = []
        for block in self.encoder_blocks:
            x, skip = block(x, t)
            shortcuts.append(skip)

        x = self.mid_block(x)

        for block, skip in zip(self.decoder_blocks, reversed(shortcuts)):
            x = block(x, skip, t)

        return self.final_conv(x)

    @staticmethod
    def _compute_channel_pairs(base_dim, dim_mults):
        """Build list of (in_channels, out_channels) for each encoder stage."""
        dims = [base_dim] + [base_dim * m for m in dim_mults]
        return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


if __name__ == "__main__":
    x = torch.randn(3, 3, 224, 224)
    t = torch.randint(0, 1000, (3,))
    model = Unet(1000, 128)
    y = model(x, t)
    print(f"Output shape: {y.shape}")
