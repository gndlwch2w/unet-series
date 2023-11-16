import torch
from torch import nn
from .unet import DoubleConv, Down, OutConv

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_concat=2, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = DoubleConv(in_channels + (num_concat - 1) * out_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + (num_concat - 2) * out_channels, out_channels)

    def forward(self, X0, *Xs):
        X0 = self.up(X0)
        X = torch.concat([X0, *Xs], dim=1)
        return self.conv(X)

class UNetPlusPlus(nn.Module):
    """
    Refs:
        https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/UNet_2Plus.py
        https://github.com/ZJUGiveLab/UNet-Version/blob/master/models/layers.py
    """

    def __init__(self, n_classes=1, n_channels=3, bilinear=True, fix_weight=True):
        super(UNetPlusPlus, self).__init__()

        # Encoder
        self.down00 = DoubleConv(n_channels, 64)
        self.down10 = Down(64, 128)
        self.down20 = Down(128, 256)
        self.down30 = Down(256, 512)
        self.down40 = Down(512, 1024)

        # Decoder
        self.up01 = Up(128, 64, 2, bilinear)

        self.up11 = Up(256, 128, 2, bilinear)
        self.up02 = Up(128, 64, 3, bilinear)

        self.up21 = Up(512, 256, 2, bilinear)
        self.up12 = Up(256, 128, 3, bilinear)
        self.up03 = Up(128, 64, 4, bilinear)

        self.up31 = Up(1024, 512, 2, bilinear)
        self.up22 = Up(512, 256, 3, bilinear)
        self.up13 = Up(256, 128, 4, bilinear)
        self.up04 = Up(128, 64, 5, bilinear)

        # Output
        self.out1 = OutConv(64, n_classes)
        self.out2 = OutConv(64, n_classes)
        self.out3 = OutConv(64, n_classes)
        self.out4 = OutConv(64, n_classes)

        # Weighted deep supervision
        self.weight = torch.FloatTensor(5)
        self.weight.data.fill_(1 / 4)
        if not fix_weight:
            self.weight = nn.Parameter(self.weight)

    def forward(self, X):
        # c0
        X00 = self.down00(X)
        X10 = self.down10(X00)
        X20 = self.down20(X10)
        X30 = self.down30(X20)
        X40 = self.down40(X30)
        # c1
        X01 = self.up01(X10, X00)
        X11 = self.up11(X20, X10)
        X21 = self.up21(X30, X20)
        X31 = self.up31(X40, X30)
        # c2
        X02 = self.up02(X11, X00, X01)
        X12 = self.up12(X21, X10, X11)
        X22 = self.up22(X31, X20, X21)
        # c3
        X03 = self.up03(X12, X00, X01, X02)
        X13 = self.up13(X22, X10, X11, X12)
        # c4
        X04 = self.up04(X13, X00, X01, X02, X03)
        # out
        O1 = self.out1(X01)
        O2 = self.out1(X02)
        O3 = self.out1(X03)
        O4 = self.out1(X04)
        O = self.weight[0] * O1 + self.weight[1] * O2 + self.weight[2] * O3 + self.weight[3] * O4
        return O
