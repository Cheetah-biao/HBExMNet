import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """A compact 3D convolution block used throughout the U-Net."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, input_tensor):
        return self.conv(input_tensor)


class Unet(nn.Module):
    """
    A lightweight 3D U-Net with two encoder levels.

    The structure mirrors the original project design:
    encoder -> bottleneck -> decoder -> 1-channel prediction.
    """

    def __init__(self, num_features=32, in_ch=1, out_ch=1):
        super().__init__()

        self.conv1 = DoubleConv(in_ch, num_features)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = DoubleConv(num_features, num_features * 2)
        self.pool2 = nn.MaxPool3d(2)

        self.middle = nn.Sequential(
            nn.Conv3d(num_features * 2, num_features * 4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_features * 4, num_features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv1 = nn.Sequential(
            nn.Conv3d(num_features * 4, num_features * 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_features * 2, num_features, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.up_conv2 = DoubleConv(num_features * 2, num_features)
        self.final_conv = nn.Conv3d(num_features, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        mid = self.middle(p2)

        up1 = self.up1(mid)
        merge1 = torch.cat([up1, c2], dim=1)
        c8 = self.up_conv1(merge1)

        up2 = self.up2(c8)
        merge2 = torch.cat([up2, c1], dim=1)
        c9 = self.up_conv2(merge2)

        return self.final_conv(c9)


def model_info(model):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    print("\n%5s %50s %9s %12s %20s %12s %12s" % ("layer", "name", "gradient", "parameters", "shape", "mu", "sigma"))
    for i, (name, p) in enumerate(model.named_parameters()):
        name = name.replace("module_list.", "")
        print("%5g %50s %9s %12g %20s %12.3g %12.3g" % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    print("Model Summary: %g layers, %g parameters, %g gradients\n" % (i + 1, n_p, n_g))


if __name__ == "__main__":
    net = Unet(num_features=32)
    model_info(net)

    x = torch.randn(2, 1, 32, 32, 32)
    y = net(x)
    print("Output shape:", y.shape)
