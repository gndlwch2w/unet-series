import warnings
import torch
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    @staticmethod
    def forward(ctx, *args):
        input, target = args
        eps = 0.0001
        inter = torch.dot(input.view(-1), target.view(-1))
        union = torch.sum(input) + torch.sum(target) + eps
        ctx.save_for_backward(input, target, inter, union)
        t = (2 * inter.float() + eps) / union.float()
        return t

    @staticmethod
    def backward(ctx, *grad_outputs):
        grad_output, = grad_outputs
        input, target, inter, union = ctx.saved_tensors
        grad_input = grad_target = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output * 2 * \
                (target * union - inter) / (union * union)
        if ctx.needs_input_grad[1]:
            warnings.warn(
                "This function has only a single output, so it gets only one gradient")
            grad_target = None
        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()
    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff.apply(c[0], c[1])
    return s / input.shape[0]
