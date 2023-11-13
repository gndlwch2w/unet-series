from torch import nn
import torch
from model import UNet
from data import *
from torch.utils.data import DataLoader
from unittest import TestCase

dir_valid_imgs = 'E:/ccw/dataset/PH2/images/'
dir_valid_labels = 'E:/ccw/dataset/PH2/masks/'


class Test(TestCase):
    def test_train_net(self):
        tsf = Compose([
            Resize(512, 512),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        dataset = ISIC2018Dataset(dir_valid_imgs, dir_valid_labels, tsf)
        dataloader = DataLoader(dataset, batch_size=24,
                                shuffle=False, num_workers=4, pin_memory=True)
        criterion = nn.BCEWithLogitsLoss()
        net = UNet(n_channels=3, n_classes=1, bilinear=True)
        for X, y in dataloader:
            y_hat = net(X)
            print(y.shape)
            print(y_hat.shape)
            print(torch.unique(y.reshape(-1)))
            print(criterion(y_hat, y.to(torch.float32)))
            break
