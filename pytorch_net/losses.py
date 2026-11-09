import math

import torch
import torch.nn.functional as F

LN2 = math.log(2)


def cross_entropy_bits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y) / LN2


def gaussian_kl(mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
    kl_per_dim = 0.5 * (sigma**2 + mu**2 - 1) - torch.log(sigma)
    return kl_per_dim.sum(dim=1).mean()


def cholesky_gaussian_kl(mu: torch.Tensor, scale_tril: torch.Tensor) -> torch.Tensor:
    trace = scale_tril.pow(2).sum(dim=(1, 2))
    log_det = 2 * torch.log(torch.diagonal(scale_tril, dim1=1, dim2=2)).sum(dim=1)
    return (0.5 * (trace + mu.pow(2).sum(dim=1) - mu.shape[1] - log_det)).mean()


def vib_loss_bits(
    logits: torch.Tensor, y: torch.Tensor, kl: torch.Tensor, beta: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, y)
    return (ce + beta * kl) / LN2, ce / LN2, kl / LN2
