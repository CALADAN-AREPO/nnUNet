import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss
from nnunetv2.training.loss.robust_ce_loss import RobustCrossEntropyLoss


class BoundaryLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits: torch.Tensor, dist_maps: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)
        boundary_loss = (probs * dist_maps).sum(dim=(2, 3))
        return boundary_loss.mean()


def soft_boundary(probs: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    padding = kernel_size // 2
    eroded = -F.max_pool2d(-probs, kernel_size, stride=1, padding=padding)
    return probs - eroded


class ExpMASDLoss(nn.Module):
    def __init__(self, tau: float = 0.02):
        super().__init__()
        self.tau = tau

    def forward(self, logits: torch.Tensor, gt_dist_maps: torch.Tensor) -> torch.Tensor:
        H = logits.shape[2]
        probs = F.softmax(logits, dim=1)
        soft_bounds = soft_boundary(probs)
        soft_masd = (soft_bounds * gt_dist_maps).sum(dim=(2, 3))
        soft_masd = soft_masd / (soft_bounds.sum(dim=(2, 3)) + 1e-8)
        soft_masd = soft_masd / H
        score_proxy = torch.exp(-soft_masd / self.tau)
        return 1.0 - score_proxy.mean()


class DAOCTLoss(nn.Module):
    def __init__(self, tau: float = 0.02,
                 w_dice: float = 1.0,
                 w_ce: float = 1.0,
                 w_bdry: float = 0.5,
                 w_masd: float = 0.5):
        super().__init__()
        self.dice = MemoryEfficientSoftDiceLoss(apply_nonlin=nn.Softmax(dim=1), smooth=1e-5, do_bg=False, batch_dice=True, ddp=False)
        self.ce = RobustCrossEntropyLoss()
        self.boundary = BoundaryLoss()
        self.exp_masd = ExpMASDLoss(tau=tau)
        self.w_dice = w_dice
        self.w_ce = w_ce
        self.w_bdry = w_bdry
        self.w_masd = w_masd

    def forward(self, logits: torch.Tensor, seg_target: torch.Tensor,
                dist_maps: torch.Tensor = None, epoch: int = 0, max_epochs: int = 500) -> torch.Tensor:
        l_dice = self.dice(logits, seg_target)
        l_ce = self.ce(logits, seg_target[:, 0].long())

        boundary_weight = min(1.0, max(0.0, (epoch - 50) / 100))
        l_bdry = self.boundary(logits, dist_maps) if dist_maps is not None else torch.zeros_like(l_dice)
        l_masd = self.exp_masd(logits, dist_maps) if dist_maps is not None else torch.zeros_like(l_dice)

        total = (self.w_dice * l_dice
                 + self.w_ce * l_ce
                 + boundary_weight * (self.w_bdry * l_bdry
                                      + self.w_masd * l_masd))
        return total
