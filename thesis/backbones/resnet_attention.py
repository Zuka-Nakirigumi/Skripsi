from typing import Callable, Optional

import torch
import torch.nn as nn

from functools import partial

from thesis.attention.eca import EfficientChannelAttention
from thesis.attention.cbam import CBAM
from thesis.attention.procbam import ProCBAM

AttentionFactory = Callable[[int], nn.Module]


def conv3x3(
    in_channels,
    out_channels,
    stride=1,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


def conv1x1(
    in_channels,
    out_channels,
    stride=1,
):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=1,
        stride=stride,
        bias=False,
    )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        attention_factory: Optional[
            AttentionFactory
        ] = None,
    ):
        super().__init__()

        self.conv1 = conv3x3(
            in_channels,
            out_channels,
            stride=stride,
        )

        self.bn1 = nn.BatchNorm2d(
            out_channels
        )

        self.relu = nn.ReLU(
            inplace=True
        )

        self.conv2 = conv3x3(
            out_channels,
            out_channels,
        )

        self.bn2 = nn.BatchNorm2d(
            out_channels
        )

        # Sekarang nilai attention adalah Identity.
        # Nanti akan diganti menjadi ECA / CBAM / ProCBAM.
        if attention_factory is None:
            self.attention = nn.Identity()
        else:
            self.attention = (
                attention_factory(
                    out_channels
                )
            )

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Lokasi attention dikunci:
        #
        # conv2
        #   ↓
        # bn2
        #   ↓
        # attention
        #   ↓
        # residual addition
        #
        out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class RemoteSensingResNet(nn.Module):

    def __init__(
        self,
        layers,
        attention_factory=None,
        zero_init_residual=False,
    ):
        super().__init__()

        self.in_channels = 64

        # Standard ImageNet-style ResNet stem.
        self.conv1 = nn.Conv2d(
            3,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.relu = nn.ReLU(
            inplace=True
        )

        self.maxpool = nn.MaxPool2d(
            kernel_size=3,
            stride=2,
            padding=1,
        )

        self.layer1 = self._make_layer(
            out_channels=64,
            blocks=layers[0],
            stride=1,
            attention_factory=(
                attention_factory
            ),
        )

        self.layer2 = self._make_layer(
            out_channels=128,
            blocks=layers[1],
            stride=2,
            attention_factory=(
                attention_factory
            ),
        )

        self.layer3 = self._make_layer(
            out_channels=256,
            blocks=layers[2],
            stride=2,
            attention_factory=(
                attention_factory
            ),
        )

        self.layer4 = self._make_layer(
            out_channels=512,
            blocks=layers[3],
            stride=2,
            attention_factory=(
                attention_factory
            ),
        )

        # Tidak tergantung resolusi input.
        self.avgpool = (
            nn.AdaptiveAvgPool2d(
                (1, 1)
            )
        )

        self.out_dim = 512

        self._initialize_weights()

        if zero_init_residual:
            self._zero_init_residual()

    def _make_layer(
        self,
        out_channels,
        blocks,
        stride,
        attention_factory,
    ):
        downsample = None

        if (
            stride != 1
            or
            self.in_channels
            != out_channels
        ):
            downsample = nn.Sequential(
                conv1x1(
                    self.in_channels,
                    out_channels,
                    stride=stride,
                ),
                nn.BatchNorm2d(
                    out_channels
                ),
            )

        layers = [
            BasicBlock(
                in_channels=(
                    self.in_channels
                ),
                out_channels=(
                    out_channels
                ),
                stride=stride,
                downsample=downsample,
                attention_factory=(
                    attention_factory
                ),
            )
        ]

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    in_channels=(
                        self.in_channels
                    ),
                    out_channels=(
                        out_channels
                    ),
                    stride=1,
                    downsample=None,
                    attention_factory=(
                        attention_factory
                    ),
                )
            )

        return nn.Sequential(*layers)

    def _initialize_weights(self):
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

    def _zero_init_residual(self):
        for module in self.modules():

            if isinstance(
                module,
                BasicBlock,
            ):
                nn.init.constant_(
                    module.bn2.weight,
                    0,
                )

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        pooled = self.avgpool(x4)

        features = torch.flatten(
            pooled,
            1,
        )

        return {
            "fmaps": [
                x1,
                x2,
                x3,
                x4,
            ],
            "features": features,
        }

    @property
    def last_conv(self):
        """
        Dibutuhkan oleh sebagian utilitas PyCIL,
        misalnya Grad-CAM.
        """
        return self.layer4[-1].conv2


def _resolve_attention(args):
    """ Menambahkan ECA, CBAM, dan ProCBAM """
    if args is None:
        attention_type = "none"
    else:
        attention_type = str(
            args.get(
                "attention_type",
                "none",
            )
        ).lower()

    if attention_type in {
        "none",
        "identity",
        "",
    }:
        return "none", None

    if attention_type == "eca":
        gamma = float(args.get("eca_gamma",2.0))
        b = float(args.get("eca_b",1.0))
        attention_factory = partial(
            EfficientChannelAttention,
            gamma=gamma,
            b=b,
        )
        return ("eca",attention_factory)

    if attention_type == "cbam":
        reduction_ratio = int(
            args.get(
                "cbam_reduction",
                16,
            )
        )
        spatial_kernel = int(
            args.get(
                "cbam_spatial_kernel",
                7,
            )
        )
        attention_factory = partial(
            CBAM,
            reduction_ratio=(
                reduction_ratio
            ),
            spatial_kernel_size=(
                spatial_kernel
            ),
        )
        return ("cbam",attention_factory)

    if attention_type == "procbam":
        reduction_ratio = int(
            args.get(
                "procbam_reduction",
                16,
            )
        )

        spatial_kernel = int(
            args.get(
                "procbam_spatial_kernel",
                7,
            )
        )

        se_reduction = int(
            args.get(
                "procbam_se_reduction",
                16,
            )
        )

        attention_factory = partial(
            ProCBAM,
            reduction_ratio=(
                reduction_ratio
            ),
            spatial_kernel_size=(
                spatial_kernel
            ),
            se_reduction_ratio=(
                se_reduction
            ),
        )
        return ("procbam",attention_factory)

    raise NotImplementedError(
        f"Attention '{attention_type}' "
        "belum tersedia."
    )


def resnet18_remote(
    pretrained=False,
    args=None,
    **kwargs,
):
    if pretrained:
        raise ValueError(
            "Thesis backbone dilatih "
            "from scratch. "
            "pretrained harus False."
        )

    attention_type, attention_factory = (
        _resolve_attention(args)
    )

    model = RemoteSensingResNet(
        layers=[2, 2, 2, 2],
        attention_factory=(
            attention_factory
        ),
        **kwargs,
    )

    model.attention_type = (
        attention_type
    )

    return model


def resnet34_remote(
    pretrained=False,
    args=None,
    **kwargs,
):
    if pretrained:
        raise ValueError(
            "Thesis backbone dilatih "
            "from scratch. "
            "pretrained harus False."
        )

    attention_type, attention_factory = (
        _resolve_attention(args)
    )

    model = RemoteSensingResNet(
        layers=[3, 4, 6, 3],
        attention_factory=(
            attention_factory
        ),
        **kwargs,
    )

    model.attention_type = (
        attention_type
    )

    return model