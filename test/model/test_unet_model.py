import torch
from model import *
from loss import CenterLoss
from unittest import TestCase


class TestUNet(TestCase):
    def test_forward(self):
        net = UNet(3, 1, True).to("cuda:0")
        loss = CenterLoss(2)
        features, y_hat = net(torch.randn(16, 3, 512, 512).to("cuda:0"))
        labels = torch.zeros((16, 1, 512, 512)).reshape(-1).to("cuda:0")
        features = features.permute(0, 2, 3, 1).reshape(-1, features.shape[1])
        l = loss(features, labels)
        print(l)
