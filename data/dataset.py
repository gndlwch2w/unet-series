import os.path
import logging
from glob import glob
from PIL import Image
from torch.utils.data import Dataset


class ISIC2018Dataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None):
        self.imgs_dir = imgs_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.imgs = glob(os.path.join(self.imgs_dir, "*.*"))
        logging.info(f'Loading dataset with {len(self.imgs)} examples')

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_dir = self.imgs[idx]
        # labels_dir = os.path.join(self.labels_dir, f'{os.path.basename(img_dir).split(".")[0]}_segmentation.*')
        labels_dir = os.path.join(
            self.labels_dir, f'{os.path.basename(img_dir).split(".")[0]}.*')
        img = Image.open(img_dir)
        labels = Image.open(glob(labels_dir)[0]).convert('L')
        assert img.size == labels.size
        if self.transform is not None:
            img, labels = self.transform(img, labels)
        return img, labels.unsqueeze(0)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
