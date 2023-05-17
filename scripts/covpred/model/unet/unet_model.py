""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        c_1, d_1, d_2, d_3, d_4, u_1, u_2, u_3, u_4 = (
            32,
            64,
            128,
            256,
            512,
            256,
            128,
            64,
            32,
        )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, c_1)
        self.down1 = Down(c_1, d_1)
        self.down2 = Down(d_1, d_2)
        self.down3 = Down(d_2, d_3)
        factor = 2 if bilinear else 1
        self.down4 = Down(d_3, d_4 // factor)
        self.up1 = Up(d_4, u_1 // factor, bilinear)
        self.up2 = Up(u_1, u_2 // factor, bilinear)
        self.up3 = Up(u_2, u_3 // factor, bilinear)
        self.up4 = Up(u_3, u_4, bilinear)
        self.outc = OutConv(u_4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetM(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetM, self).__init__()
        c_1, d_1, d_2, d_3, d_4, u_1, u_2, u_3, u_4 = (
            16,
            32,
            64,
            128,
            256,
            128,
            64,
            32,
            16,
        )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, c_1)
        self.down1 = Down(c_1, d_1)
        self.down2 = Down(d_1, d_2)
        self.down3 = Down(d_2, d_3)
        factor = 2 if bilinear else 1
        self.down4 = Down(d_3, d_4 // factor)
        self.up1 = Up(d_4, u_1 // factor, bilinear)
        self.up2 = Up(u_1, u_2 // factor, bilinear)
        self.up3 = Up(u_2, u_3 // factor, bilinear)
        self.up4 = Up(u_3, u_4, bilinear)
        self.outc = OutConv(u_4, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class UNetSmall(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetSmall, self).__init__()
        c_1, d_1, d_2, d_3, u_3, u_2, u_1 = (
            16,
            32,
            64,
            128,
            64,
            32,
            16,
        )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, c_1)
        self.down1 = Down(c_1, d_1)
        self.down2 = Down(d_1, d_2)
        self.down3 = Down(d_2, d_3)
        factor = 2 if bilinear else 1
        self.up3 = Up(d_3, u_3 // factor, bilinear)
        self.up2 = Up(u_3, u_2 // factor, bilinear)
        self.up1 = Up(u_2, u_1, bilinear)
        self.outc = OutConv(u_1, n_classes)

    def forward(self, x):
        # print(f"Input data on {x.device}")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up3(x4, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits


class UNetXS(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNetXS, self).__init__()
        c_1, d_1, d_2, u_2, u_1 = (
            16,
            32,
            64,
            32,
            16,
        )
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, c_1)
        self.down1 = Down(c_1, d_1)
        self.down2 = Down(d_1, d_2)
        factor = 2 if bilinear else 1
        self.up2 = Up(d_2, u_2 // factor, bilinear)
        self.up1 = Up(u_2, u_1, bilinear)
        self.outc = OutConv(u_1, n_classes)

    def forward(self, x):
        # print(f"Input data on {x.device}")
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up2(x3, x2)
        x = self.up1(x, x1)
        logits = self.outc(x)
        return logits
