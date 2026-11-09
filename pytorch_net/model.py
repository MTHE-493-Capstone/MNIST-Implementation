import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import cholesky_gaussian_kl, gaussian_kl

Z_DIM = 256
SIGMA_SHIFT = 5.0
OFF_DIAG_SCALE = 1e-2


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
    def __init__(self, z_dim: int = Z_DIM, encoder_outputs: int | None = None) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, encoder_outputs or 2 * z_dim),
        )
        self.decoder = nn.Linear(z_dim, 10)
        xavier_init(self)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, sigma_raw = self.encoder(x).chunk(2, dim=1)
        return mu, F.softplus(sigma_raw - SIGMA_SHIFT)

    def sample(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return mu + sigma * torch.randn_like(sigma)

    def kl(self, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return gaussian_kl(mu, sigma)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, scale = self.encode(x)
        return self.decoder(self.sample(mu, scale)), mu, scale


class VIB2D(VIB):
    def __init__(self) -> None:
        super().__init__(z_dim=2, encoder_outputs=6)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.encoder(x)
        l11, _upper, l21, l22 = out[:, 2:].unbind(dim=1)
        diag = F.softplus(torch.stack([l11, l22], dim=1) - SIGMA_SHIFT)
        scale_tril = torch.stack(
            [
                torch.stack([diag[:, 0], torch.zeros_like(l11)], dim=1),
                torch.stack([OFF_DIAG_SCALE * l21, diag[:, 1]], dim=1),
            ],
            dim=1,
        )
        return out[:, :2], scale_tril

    def sample(self, mu: torch.Tensor, scale_tril: torch.Tensor) -> torch.Tensor:
        return mu + (scale_tril @ torch.randn_like(mu).unsqueeze(-1)).squeeze(-1)

    def kl(self, mu: torch.Tensor, scale_tril: torch.Tensor) -> torch.Tensor:
        return cholesky_gaussian_kl(mu, scale_tril)
