from collections.abc import Iterator
from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent.parent / "data" / "mnist.pt"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize(X: torch.Tensor) -> torch.Tensor:
    X = 2 * (X.to(torch.float32) / 255) - 1
    assert X.min() >= -1 and X.max() <= 1
    return X


def load_mnist(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d = {k: v.to(device) for k, v in torch.load(DATA_PATH).items()}
    return normalize(d["X_train"]), d["y_train"], normalize(d["X_test"]), d["y_test"]


def batches(X: torch.Tensor, y: torch.Tensor, batch_size: int = 100) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    perm = torch.randperm(len(X), device=X.device)
    for i in range(0, len(X), batch_size):
        idx = perm[i : i + batch_size]
        yield X[idx], y[idx]
