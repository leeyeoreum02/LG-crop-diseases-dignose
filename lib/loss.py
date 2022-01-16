import torch
from torch import nn
import torch.nn.functional as F


def smooth(y, num_classes, smooth_factor):
    y = y.float()
    # assert len(y.shape) == 2
    y *= 1 - smooth_factor
    y += smooth_factor / num_classes
    return y


class SigmoidFocalCrossEntropyLoss(nn.Module):
    def __init__(
        self, 
        alpha=0.25, 
        gamma=2.0, 
        factor=0.1, 
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.factor = factor
        
    def forward(self, y_pred, y_true):
        # y_true = F.one_hot(y_true, num_classes=y_pred.shape[-1]).float()
        y_true = smooth(y_true, self.factor)
        ce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        return torch.sum(alpha_factor * modulating_factor * ce, axis=-1)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(self, weight=None, gamma=2, factor=0.1, reduction='mean'):
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.factor = factor

    def forward(self, input, target):
        # num_classes = input.shape[-1]
        # target = smooth(target.float(), num_classes, self.factor)
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
