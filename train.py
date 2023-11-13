import logging
import os
import sys
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
from eval import eval_net
from model import UNet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from data import *
from loss import dice_coeff
from loss import CenterLoss
import itertools
from loss import WeightedBCELoss

dir_train_imgs = 'E:/ccw/dataset/ISIC2017/images/'
dir_train_labels = 'E:/ccw/dataset/ISIC2017/masks'
dir_valid_imgs = 'E:/ccw/dataset/PH2/images/'
dir_valid_labels = 'E:/ccw/dataset/PH2/masks/'
dir_checkpoint = 'checkpoints/'

tsf = Compose([
    Resize(512, 512),
    RandomHorizontalFlip(0.5),
    RandomVerticalFlip(0.5),
    ToTensor(),
    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def train_net(net, device, epochs, batch_size, lr, save_cp=True):
    train_dataset = ISIC2018Dataset(dir_train_imgs, dir_train_labels, tsf)
    valid_dataset = ISIC2018Dataset(dir_valid_imgs, dir_valid_labels, tsf)
    n_train = len(train_dataset)
    n_valid = len(valid_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size,
                              shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_valid}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    center_loss = CenterLoss(2, 64, True)
    optimizer_center_loss = torch.optim.Adam(center_loss.parameters(), lr=0.5)
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        # pos_weight=torch.FloatTensor([0.8061 / 0.1939]).to(device)
        criterion = nn.BCEWithLogitsLoss()
        # criterion = WeightedBCELoss(0.8061)

    for epoch in range(epochs):
        net.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for imgs, masks in train_loader:
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                masks = masks.to(device=device, dtype=mask_type)

                features, masks_pred = net(imgs)
                # loss = criterion(masks_pred, masks)
                # loss = criterion(masks_pred, masks) - dice_coeff(masks_pred, masks)

                # Center Loss
                gamma = 0.0001
                labels = masks.view(-1)
                features = features.permute(
                    0, 2, 3, 1).reshape(-1, features.shape[1])
                bl = criterion(masks_pred, masks)
                cl = center_loss(features, labels)
                loss = bl + gamma * cl

                writer.add_scalar('Loss/total', loss.item(), global_step)
                writer.add_scalar('Loss/center loss', cl.item(), global_step)
                writer.add_scalar('Loss/cross entropy', bl.item(), global_step)
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                optimizer_center_loss.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()
                optimizer_center_loss.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                if global_step % (n_train // (10 * batch_size)) == 0:
                    # for tag, value in net.named_parameters():
                    #     tag = tag.replace('.', '/')
                    #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                    val_score = eval_net(net, valid_loader, device)
                    scheduler.step(val_score)
                    writer.add_scalar(
                        'learning_rate', optimizer.param_groups[0]['lr'], global_step)

                    if net.n_classes > 1:
                        logging.info(
                            'Validation cross entropy: {}'.format(val_score))
                        writer.add_scalar('Loss/test', val_score, global_step)
                    else:
                        logging.info(
                            'Validation Dice Coeff: {}'.format(val_score))
                        writer.add_scalar('Dice/test', val_score, global_step)

                    writer.add_images('images', imgs, global_step)
                    if net.n_classes == 1:
                        writer.add_images('masks/true', masks, global_step)
                        writer.add_images(
                            'masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N
    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    # dir_model = ''
    # net.load_state_dict(torch.load(dir_model, map_location=device))
    # logging.info(f'Model loaded from {dir_model}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=400,
                  batch_size=16,
                  lr=1e-3,
                  device=device)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
