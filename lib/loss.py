from typing import Optional, Union

import torch
from torch import nn, Tensor
import torch.nn.functional as F


def smooth(y: Tensor, num_classes: int, smooth_factor: float) -> Tensor:
    y = y.float()
    # assert len(y.shape) == 2
    y *= 1 - smooth_factor
    y += smooth_factor / num_classes
    return y


class SigmoidFocalCrossEntropyLoss(nn.Module):
    def __init__(
        self, 
        alpha: float = 0.25, 
        gamma: float = 2.0, 
        factor: float = 0.1, 
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.factor = factor
        
    def forward(self, y_pred: Tensor, y_true: Tensor) -> Tensor:
        # y_true = F.one_hot(y_true, num_classes=y_pred.shape[-1]).float()
        y_true = smooth(y_true, self.factor)
        ce = F.binary_cross_entropy_with_logits(y_pred, y_true)
        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = torch.pow((1.0 - p_t), self.gamma)

        return torch.sum(alpha_factor * modulating_factor * ce, axis=-1)


class FocalLoss(nn.modules.loss._WeightedLoss):
    def __init__(
        self, weight: Optional[Tensor] = None, gamma: int = 2, factor: float = 0.1, reduction: str = 'mean'
    ) -> None:
        super(FocalLoss, self).__init__(weight, reduction=reduction)
        self.gamma = gamma
        self.weight = weight  # weight parameter will act as the alpha parameter to balance class weights
        self.factor = factor

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # num_classes = input.shape[-1]
        # target = smooth(target.float(), num_classes, self.factor)
        ce_loss = F.cross_entropy(input, target, reduction=self.reduction, weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss
    
    
def log_t(u: Tensor, t: Tensor) -> Union[Tensor, float]:
    """Compute log_t for `u`."""

    if t == 1.0:
        return torch.log(u)
    else:
        return (u ** (1.0 - t) - 1.0) / (1.0 - t)


def exp_t(u: Tensor, t: Tensor) -> Tensor:
    """Compute exp_t for `u`."""

    if t == 1.0:
        return torch.exp(u)
    else:
        return torch.relu(1.0 + (1.0 - t) * u) ** (1.0 / (1.0 - t))


def compute_normalization_fixed_point(activations: Tensor, t: Tensor, num_iters: int = 5) -> Tensor:
    """Returns the normalization value for each example (t > 1.0).
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (> 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    mu = torch.max(activations, dim=-1).values.view(-1, 1)
    normalized_activations_step_0 = activations - mu

    normalized_activations = normalized_activations_step_0
    i = 0
    while i < num_iters:
        i += 1
        logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)
        normalized_activations = normalized_activations_step_0 * (logt_partition ** (1.0 - t))

    logt_partition = torch.sum(exp_t(normalized_activations, t), dim=-1).view(-1, 1)

    return -log_t(1.0 / logt_partition, t) + mu


def compute_normalization(activations: Tensor, t: Tensor, num_iters: int = 5) -> Optional[Tensor]:
    """Returns the normalization value for each example.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature 2 (< 1.0 for finite support, > 1.0 for tail heaviness).
    num_iters: Number of iterations to run the method.
    Return: A tensor of same rank as activation with the last dimension being 1.
    """

    if t < 1.0:
        return None  # not implemented as these values do not occur in the authors experiments...
    else:
        return compute_normalization_fixed_point(activations, t, num_iters)


def tempered_softmax(activations: Tensor, t: Tensor, num_iters: int = 5) -> Tensor:
    """Tempered softmax function.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    t: Temperature tensor > 0.0.
    num_iters: Number of iterations to run the method.
    Returns:
    A probabilities tensor.
    """

    if t == 1.0:
        normalization_constants = torch.log(torch.sum(torch.exp(activations), dim=-1))
    else:
        normalization_constants = compute_normalization(activations, t, num_iters)

    return exp_t(activations - normalization_constants, t)

 
def bi_tempered_logistic_loss(
    activations: Tensor, 
    labels: Tensor, 
    t1: float, 
    t2: float, 
    label_smoothing: float = 0.0, 
    num_iters: int = 5
) -> Tensor:
    """Bi-Tempered Logistic Loss with custom gradient.
    Args:
    activations: A multi-dimensional tensor with last dimension `num_classes`.
    labels: A tensor with shape and dtype as activations.
    t1: Temperature 1 (< 1.0 for boundedness).
    t2: Temperature 2 (> 1.0 for tail heaviness, < 1.0 for finite support).
    label_smoothing: Label smoothing parameter between [0, 1).
    num_iters: Number of iterations to run the method.
    Returns:
    A loss tensor.
    """
    labels = F.one_hot(labels, num_classes=25)
    if label_smoothing > 0.0:
        num_classes = labels.shape[-1]
        labels = (1 - num_classes / (num_classes - 1) * label_smoothing) * labels + label_smoothing / (num_classes - 1)

    probabilities = tempered_softmax(activations, t2, num_iters)

    temp1 = (log_t(labels + 1e-10, t1) - log_t(probabilities, t1)) * labels
    temp2 = (1 / (2 - t1)) * (torch.pow(labels, 2 - t1) - torch.pow(probabilities, 2 - t1))
    loss_values = temp1 - temp2

    return torch.sum(loss_values, dim=-1)


class BiTemperedLogisticLoss:
    def __init__(self, t1: float, t2: float, label_smoothing: float = 0.0) -> None:
        self.t1 = t1
        self.t2 = t2
        self.label_smoothing = label_smoothing
        
    def __call__(self, inputs: Tensor, targets: Tensor) -> Tensor:
        losses = bi_tempered_logistic_loss(
            inputs, targets, self.t1, self.t2, self.label_smoothing
        )
        return losses.mean()
