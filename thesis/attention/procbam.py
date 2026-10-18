import torch
import torch.nn as nn

from thesis.attention.cbam import (
    ChannelAttention,
    SpatialAttention,
)


class SqueezeExcitation(nn.Module):
    """
    Squeeze-and-Excitation module.

    GAP
     ↓
    FC: C -> C/r
     ↓
    ReLU
     ↓
    FC: C/r -> C
     ↓
    Sigmoid
     ↓
    channel-wise multiplication
    """

    def __init__(
        self,
        channels,
        reduction_ratio=16,
    ):
        super().__init__()

        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )

        if reduction_ratio <= 0:
            raise ValueError(
                "reduction_ratio must be positive."
            )

        hidden_channels = max(
            channels // reduction_ratio,
            1,
        )

        self.channels = channels
        self.reduction_ratio = (
            reduction_ratio
        )

        self.avg_pool = (
            nn.AdaptiveAvgPool2d(1)
        )

        self.fc = nn.Sequential(
            nn.Linear(
                channels,
                hidden_channels,
            ),

            nn.ReLU(),

            nn.Linear(
                hidden_channels,
                channels,
            ),

            nn.Sigmoid(),
        )

    def forward(self, x):

        batch_size, channels, _, _ = (
            x.shape
        )

        # [B, C, H, W]
        #      ↓
        # [B, C]
        y = self.avg_pool(x)

        y = y.view(
            batch_size,
            channels,
        )

        y = self.fc(y)

        # [B, C]
        #   ↓
        # [B, C, 1, 1]
        y = y.view(
            batch_size,
            channels,
            1,
            1,
        )

        return x * y


class ProCBAM(nn.Module):
    """
    Parallel Convolutional Block Attention Module.

    CAM dan SAM bekerja secara parallel pada
    input feature map yang sama.

                ┌── CAM ──┐
        x ──────┤         ├── Add ── SE
                └── SAM ──┘
    """

    def __init__(
        self,
        channels,
        reduction_ratio=16,
        spatial_kernel_size=7,
        se_reduction_ratio=16,
    ):
        super().__init__()

        self.channels = channels

        self.reduction_ratio = (
            reduction_ratio
        )

        self.spatial_kernel_size = (
            spatial_kernel_size
        )

        self.se_reduction_ratio = (
            se_reduction_ratio
        )

        # Branch 1: Channel attention
        self.channel_attention = (
            ChannelAttention(
                channels=channels,
                reduction_ratio=(
                    reduction_ratio
                ),
            )
        )

        # Branch 2: Spatial attention
        self.spatial_attention = (
            SpatialAttention(
                kernel_size=(
                    spatial_kernel_size
                ),
            )
        )

        # Refinement setelah kedua branch
        # digabungkan.
        self.squeeze_excitation = (
            SqueezeExcitation(
                channels=channels,
                reduction_ratio=(
                    se_reduction_ratio
                ),
            )
        )

    def forward(self, x):

        # PENTING:
        # CAM dan SAM sama-sama menerima x asli.
        channel_out = (self.channel_attention(x))
        spatial_out = (self.spatial_attention(x))

        # Pixel-wise addition sesuai
        # struktur ProCBAM.
        fused = (channel_out+ spatial_out)

        # Rapikan kembali hubungan channel
        # setelah fusion.
        out = (self.squeeze_excitation(fused))

        return out