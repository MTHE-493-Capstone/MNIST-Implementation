import math

import torch
import torch.nn.functional as F


def cross_entropy_bits(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, y) / math.log(2)
