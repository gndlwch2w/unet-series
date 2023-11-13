import torch
from model import UNet as _UNet


def _load_dict(net, url: str):
    is_remote = url.startswith("http")
    if is_remote:
        state_dict = torch.hub.load_state_dict_from_url(url, progress=True)
    else:
        state_dict = torch.load(url)
    net.load_state_dict(state_dict)
    return net


def unet_carvana(pretrained=False):
    """
    UNet model trained on the Carvana dataset ( https://www.kaggle.com/c/carvana-image-masking-challenge/data ).
    Set the scale to 1 (100%) when predicting.
    """
    net = _UNet(n_channels=3, n_classes=1, bilinear=True)
    if pretrained:
        url = 'https://github.com/milesial/Pytorch-UNet/releases/download/v1.0/unet_carvana_scale1_epoch5.pth'
        return _load_dict(net, url)
    return net


def unet_isic2017(pretrained=True):
    net = _UNet(n_channels=3, n_classes=1, bilinear=True)
    if pretrained:
        path = "pretrained_model/ISIC2017_DICE_0.76_BS_16_LR_1e-3_EPOCH_5_SIZE_512.pth"
        return _load_dict(net, path)
    return net
