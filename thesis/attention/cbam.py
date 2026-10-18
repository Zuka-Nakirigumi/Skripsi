import torch
import torch.nn as nn


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0),-1)


class ChannelAttention(nn.Module):

    def __init__(self,channels,reduction_ratio=16):
        super().__init__()

        if channels <= 0:
            raise ValueError("channels must be positive.")

        if reduction_ratio <= 0:
            raise ValueError("reduction_ratio must be positive.")

        hidden_channels = max(channels // reduction_ratio,1)

        self.channels = channels
        self.reduction_ratio = (reduction_ratio)

        self.avg_pool = (nn.AdaptiveAvgPool2d(1))

        self.max_pool = (nn.AdaptiveMaxPool2d(1))

        # Shared MLP untuk avg dan max pooling.
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channels,hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels,channels)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))

        max_out = self.mlp(self.max_pool(x))

        attention = (avg_out + max_out)

        attention = (self.sigmoid(attention))

        attention = (
            attention
            .unsqueeze(-1)
            .unsqueeze(-1)
        )
        return x * attention

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        if kernel_size not in {3,7}:
            raise ValueError("Spatial kernel must be 3 or 7.")

        padding = (
            kernel_size - 1
        ) // 2

        self.kernel_size = (kernel_size)

        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False,
        )

        # Implementasi referensi CBAM
        # memakai BN pada spatial gate.
        self.bn = nn.BatchNorm2d(
            1,
            eps=1e-5,
            momentum=0.01,
            affine=True,
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_out = torch.mean(
            x,
            dim=1,
            keepdim=True,
        )

        max_out, _ = torch.max(
            x,
            dim=1,
            keepdim=True,
        )

        spatial = torch.cat(
            [
                max_out,
                avg_out,
            ],
            dim=1,
        )

        spatial = self.conv(
            spatial
        )

        spatial = self.bn(
            spatial
        )

        spatial = self.sigmoid(
            spatial
        )

        return x * spatial


class CBAM(nn.Module):
    def __init__(self,channels,reduction_ratio=16,spatial_kernel_size=7):
        super().__init__()

        self.channel_attention = (ChannelAttention(channels=channels,reduction_ratio=(reduction_ratio)))
        self.spatial_attention = (SpatialAttention(kernel_size=(spatial_kernel_size)))

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x