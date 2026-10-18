import torch
from torch import nn
from torch.nn import functional as F

from thesis.attention.eca import (
    EfficientChannelAttention,
)

from thesis.attention.cbam import (
    CBAM,
)

from thesis.attention.procbam import (
    ProCBAM,
)


def _make_attention(
    channels,
    args,
):
    attention_type = str(
        args.get(
            "attention_type",
            "none",
        )
    ).lower()

    if attention_type == "none":

        return nn.Identity()

    if attention_type == "eca":

        return EfficientChannelAttention(
            channels=channels,
            gamma=float(
                args.get(
                    "eca_gamma",
                    2.0,
                )
            ),
            b=float(
                args.get(
                    "eca_b",
                    1.0,
                )
            ),
        )

    if attention_type == "cbam":

        return CBAM(
            channels=channels,
            reduction=int(
                args.get(
                    "cbam_reduction",
                    16,
                )
            ),
            spatial_kernel=int(
                args.get(
                    "cbam_spatial_kernel",
                    7,
                )
            ),
        )

    if attention_type == "procbam":

        return ProCBAM(
            channels=channels,
            reduction=int(
                args.get(
                    "procbam_reduction",
                    16,
                )
            ),
            spatial_kernel=int(
                args.get(
                    "procbam_spatial_kernel",
                    7,
                )
            ),
            se_reduction=int(
                args.get(
                    "procbam_se_reduction",
                    16,
                )
            ),
        )

    raise ValueError(
        f"Unknown attention_type: "
        f"{attention_type}"
    )


class SSREConvBlock(nn.Module):
    """
    Main 3x3 convolution +
    parallel 1x1 SSRE adapter.

    Adapter selalu dibuat karena dibutuhkan
    kembali pada incremental tasks.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        mode="parallel_adapters",
    ):
        super().__init__()

        self.mode = mode

        # Preserve canonical ResNet:
        # no convolution bias.
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

        self.adapter = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )


    def forward(
        self,
        x,
    ):
        out = self.conv(x)

        if (
            self.mode
            == "parallel_adapters"
        ):
            out = (
                out
                + self.adapter(x)
            )

        return out


class SSREBasicBlock(nn.Module):

    expansion = 1


    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
        args,
    ):
        super().__init__()

        mode = args.get(
            "mode",
            "parallel_adapters",
        )

        self.conv1 = SSREConvBlock(
            in_channels,
            out_channels,
            stride=stride,
            mode=mode,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = SSREConvBlock(
            out_channels,
            out_channels,
            stride=1,
            mode=mode,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        self.attention = (
            _make_attention(
                out_channels,
                args,
            )
        )

        if (
            stride != 1
            or
            in_channels
            != out_channels
        ):

            self.downsample = (
                nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(
                        out_channels
                    ),
                )
            )

        else:

            self.downsample = None


    def forward(
        self,
        x,
    ):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Same attention location
        # as our validated backbone.
        out = self.attention(out)

        if self.downsample is not None:

            identity = (
                self.downsample(x)
            )

        out = out + identity
        out = self.relu(out)

        return out


class SSREResNet(nn.Module):

    def __init__(
        self,
        layers,
        args,
    ):
        super().__init__()

        self.in_channels = 64
        self.out_dim = 512

        # Canonical ImageNet stem,
        # identical for AID/NWPU.
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(
            64
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            64,
            layers[0],
            stride=1,
            args=args,
        )

        self.layer2 = self._make_layer(
            128,
            layers[1],
            stride=2,
            args=args,
        )

        self.layer3 = self._make_layer(
            256,
            layers[2],
            stride=2,
            args=args,
        )

        self.layer4 = self._make_layer(
            512,
            layers[3],
            stride=2,
            args=args,
        )

        self.avgpool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self._initialize_weights()


    def _make_layer(
        self,
        channels,
        blocks,
        stride,
        args,
    ):
        layers = []

        layers.append(
            SSREBasicBlock(
                self.in_channels,
                channels,
                stride,
                args,
            )
        )

        self.in_channels = (
            channels
        )

        for _ in range(
            1,
            blocks,
        ):

            layers.append(
                SSREBasicBlock(
                    self.in_channels,
                    channels,
                    1,
                    args,
                )
            )

        return nn.Sequential(
            *layers
        )


    def _initialize_weights(
        self,
    ):
        for module in self.modules():

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):

                nn.init.constant_(
                    module.weight,
                    1,
                )

                nn.init.constant_(
                    module.bias,
                    0,
                )


    def switch(
        self,
        mode,
    ):
        if mode not in {
            "normal",
            "parallel_adapters",
        }:

            raise ValueError(
                f"Unknown SSRE mode: "
                f"{mode}"
            )

        for module in (
            self.modules()
        ):

            if isinstance(
                module,
                SSREConvBlock,
            ):

                module.mode = mode


    def forward(
        self,
        x,
    ):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)

        features = self.avgpool(
            f4
        )

        features = torch.flatten(
            features,
            1,
        )

        return {
            "fmaps": [
                f1,
                f2,
                f3,
                f4,
            ],
            "features":
                features,
        }


    @property
    def last_conv(
        self,
    ):
        return (
            self
            .layer4[-1]
            .conv2
            .conv
        )


def remote_ssre_resnet18(
    args,
):
    return SSREResNet(
        layers=[
            2,
            2,
            2,
            2,
        ],
        args=args,
    )


def remote_ssre_resnet34(
    args,
):
    return SSREResNet(
        layers=[
            3,
            4,
            6,
            3,
        ],
        args=args,
    )