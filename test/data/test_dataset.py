from data import *
from torch.utils.data import DataLoader
from unittest import TestCase
from tqdm import tqdm
from model import UNet

dir_train_imgs = 'E:/ccw/dataset/ISIC2017/images/'
dir_train_labels = 'E:/ccw/dataset/ISIC2017/masks'
dir_valid_imgs = 'E:/ccw/dataset/PH2/images/'
dir_valid_labels = 'E:/ccw/dataset/PH2/masks/'


class TestISIC2018Dataset(TestCase):
    def test(self):
        tsf = Compose([
            Resize(512, 512),
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        train_dataset = ISIC2018Dataset(dir_train_imgs, dir_train_labels, tsf)
        valid_dataset = ISIC2018Dataset(dir_valid_imgs, dir_valid_labels, tsf)
        train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True, num_workers=8, pin_memory=True,
                                  drop_last=True)
        valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True,
                                  drop_last=True)
        net = UNet(n_channels=3, n_classes=1, bilinear=True).to("cuda:0")

        num_positive = 0
        num_negative = 0
        with tqdm(total=len(train_dataset), unit='img') as pbar:
            for X, y in train_dataset:
                print(y.shape)
                break
                n_p = y.sum()
                num_positive += n_p
                num_negative += (262144 - n_p)
                pbar.update()
        sum_samples = num_negative + num_positive
        assert sum_samples == 262144 * 2000
        print(num_positive / sum_samples)
        print(num_negative / sum_samples)
        # tensor(0.1939)
        # tensor(0.8061)
