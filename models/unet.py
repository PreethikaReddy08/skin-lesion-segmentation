import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """
    Two consecutive convolution blocks:
    Conv -> BatchNorm -> ReLU
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """
    Baseline U-Net for binary skin lesion segmentation.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
    ) -> None:
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2,
        )

        self.encoder1 = DoubleConv(
            in_channels,
            64,
        )
        self.encoder2 = DoubleConv(
            64,
            128,
        )
        self.encoder3 = DoubleConv(
            128,
            256,
        )
        self.encoder4 = DoubleConv(
            256,
            512,
        )

        self.bottleneck = DoubleConv(
            512,
            1024,
        )

        self.upconv4 = nn.ConvTranspose2d(
            1024,
            512,
            kernel_size=2,
            stride=2,
        )
        self.decoder4 = DoubleConv(
            1024,
            512,
        )

        self.upconv3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2,
        )
        self.decoder3 = DoubleConv(
            512,
            256,
        )

        self.upconv2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2,
        )
        self.decoder2 = DoubleConv(
            256,
            128,
        )

        self.upconv1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2,
        )
        self.decoder1 = DoubleConv(
            128,
            64,
        )

        self.output_layer = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(
            self.pool(enc1)
        )
        enc3 = self.encoder3(
            self.pool(enc2)
        )
        enc4 = self.encoder4(
            self.pool(enc3)
        )

        bottleneck = self.bottleneck(
            self.pool(enc4)
        )

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat(
            (dec4, enc4),
            dim=1,
        )
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat(
            (dec3, enc3),
            dim=1,
        )
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat(
            (dec2, enc2),
            dim=1,
        )
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat(
            (dec1, enc1),
            dim=1,
        )
        dec1 = self.decoder1(dec1)

        return self.output_layer(dec1)


def test_model() -> None:
    model = UNet(
        in_channels=3,
        out_channels=1,
    )

    sample_input = torch.randn(
        1,
        3,
        256,
        256,
    )

    with torch.no_grad():
        sample_output = model(
            sample_input
        )

    print("U-Net model test")
    print("----------------")
    print(
        f"Input shape: "
        f"{tuple(sample_input.shape)}"
    )
    print(
        f"Output shape: "
        f"{tuple(sample_output.shape)}"
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Total parameters: "
        f"{parameter_count:,}"
    )
    print(
        f"Trainable parameters: "
        f"{trainable_parameter_count:,}"
    )

    assert sample_output.shape == (
        1,
        1,
        256,
        256,
    ), (
        "Expected model output shape "
        "(1, 1, 256, 256)."
    )

    print(
        "U-Net model test completed successfully."
    )


if __name__ == "__main__":
    test_model()