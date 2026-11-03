import torch
import torch.nn as nn

LR = 1e-4
BETAS = (0.5, 0.999)
LR_STEP_EPOCHS = 2
LR_GAMMA = 0.97


def make_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)


def make_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)
