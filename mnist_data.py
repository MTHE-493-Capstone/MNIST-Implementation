from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent / "data" / "mnist.pt"


def scale(X: torch.Tensor) -> torch.Tensor:
    X = X.float() / 255
    assert X.min() >= 0 and X.max() <= 1
    return X


def load_mnist() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d = torch.load(DATA_PATH)
    return scale(d["X_train"]), d["y_train"], scale(d["X_test"]), d["y_test"]
