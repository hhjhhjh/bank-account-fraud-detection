import math

import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    """Mean binary focal loss on logits and 0/1 targets, without class weights."""

    def __init__(self, gamma: float):
        super().__init__()
        if type(gamma) not in (int, float) or not math.isfinite(gamma) or gamma < 0:
            raise ValueError("training.focal_gamma must be a finite nonnegative number")
        self.gamma = float(gamma)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        correct_class_logits = (2 * targets - 1) * logits
        # Compute -log(p_t) and log(1 - p_t) directly for confident predictions.
        cross_entropy = F.softplus(-correct_class_logits)
        log_error_probability = -F.softplus(correct_class_logits)
        return (torch.exp(self.gamma * log_error_probability) * cross_entropy).mean()
