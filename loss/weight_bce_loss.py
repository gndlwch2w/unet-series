import torch
from torch import nn

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, input, target):
        # input = input.permute(0, 2, 3, 1).reshape(-1, input.shape[1])
        # input = F.sigmoid(input)
        # target = target.reshape(-1)
        # l = -1 * (self.pos_weight * target * torch.log(input) +
        #           (1 - self.pos_weight) * (1 - target) * torch.log(1 - input))
        # return l.mean()

        input = torch.sigmoid(input)
        l = -1 * (self.pos_weight * target * torch.log(input) +
                  (1 - self.pos_weight) * (1 - target) * torch.log(1 - input))
        return l.mean()
