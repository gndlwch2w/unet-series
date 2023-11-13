import torch
import torch.nn.functional as F
from tqdm import tqdm
from loss import dice_coeff


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for imgs, labels in loader:
            imgs = imgs.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=mask_type)

            with torch.no_grad():
                _, mask_pred = net(imgs)

            if net.n_classes > 1:
                tot += F.cross_entropy(mask_pred, labels).item()
            else:
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                tot += dice_coeff(pred, labels).item()
            pbar.update()

    net.train()
    return tot / n_val
