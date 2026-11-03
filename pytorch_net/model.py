import torch
import torch.nn as nn
import torch.nn.functional as F

Z_DIM = 256
SIGMA_SHIFT = 5.0


def xavier_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 10),
        )
        xavier_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class VIB(nn.Module):
    def __init__(self, z_dim: int = Z_DIM) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2 * z_dim),
        )
        self.decoder = nn.Linear(z_dim, 10)
        xavier_init(self)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, sigma_raw = self.encoder(x).chunk(2, dim=1)
        return mu, F.softplus(sigma_raw - SIGMA_SHIFT)

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu + sigma * torch.randn_like(sigma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, sigma = self.encode(x)
        return self.decoder(self.sample(mu, sigma)), mu, sigma
