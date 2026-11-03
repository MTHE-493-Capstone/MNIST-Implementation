import torch
import torch.nn as nn

LR = 1e-4
BETAS = (0.5, 0.999)


def make_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)
