import math

import torch
import torch.nn as nn


class EfficientChannelAttention(nn.Module):
    """
    Efficient Channel Attention (ECA).
    Channel attention tanpa dimensionality reduction.
    Default:
        gamma = 2
        b = 1
    Kernel size dihitung secara adaptif berdasarkan
    jumlah channel input.
    """

    def __init__(self,channels,gamma=2.0,b=1.0,kernel_size=None):
        super().__init__()
        if channels <= 0:
            raise ValueError(
                "channels must be positive."
            )
        if kernel_size is None:
            kernel_size = (
                self._adaptive_kernel_size(
                    channels=channels,
                    gamma=gamma,
                    b=b,
                )
            )
        if kernel_size <= 0:
            raise ValueError(
                "kernel_size must be positive."
            )
        if kernel_size % 2 == 0:
            raise ValueError(
                "ECA kernel_size must be odd."
            )
        self.channels = channels
        self.gamma = gamma
        self.b = b
        self.kernel_size = kernel_size
        self.avg_pool = (nn.AdaptiveAvgPool2d(1))
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(
                kernel_size - 1
            ) // 2,
            bias=False,
        )

        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def _adaptive_kernel_size(
        channels,
        gamma,
        b,
    ):
        """ Adaptive ECA kernel. k dibuat ganjil agar ukuran output Conv1D
        tetap sama dengan jumlah channel. """
        t = int(
            abs(
                (
                    math.log2(channels)
                    + b
                )
                / gamma
            )
        )

        k = (
            t
            if t % 2 == 1
            else t + 1
        )

        return max(k, 1)

    def forward(self, x):

        # [B, C, H, W]
        y = self.avg_pool(x)

        # [B, C, 1, 1]
        #
        # squeeze:
        # [B, C, 1]
        #
        # transpose:
        # [B, 1, C]
        y = (y.squeeze(-1).transpose(-1, -2))

        # Local cross-channel interaction
        y = self.conv(y)

        # [B, C, 1, 1]
        y = (y.transpose(-1, -2).unsqueeze(-1))
        y = self.sigmoid(y)

        return x * y