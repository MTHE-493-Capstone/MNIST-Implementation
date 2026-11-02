from pathlib import Path

import torch
from torchvision.datasets import MNIST

DATA_DIR = Path(__file__).parent / "data"
OUT_PATH = DATA_DIR / "mnist.pt"


def load_split(train: bool) -> tuple[torch.Tensor, torch.Tensor]:
    ds = MNIST(root=DATA_DIR / "raw", train=train, download=True)
    X = ds.data.reshape(len(ds), 28 * 28)
    y = ds.targets.to(torch.int64)
    assert X.dtype == torch.uint8
    return X, y


def main() -> None:
    X_train, y_train = load_split(train=True)
    X_test, y_test = load_split(train=False)

    torch.save(
        {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test},
        OUT_PATH,
    )

    print(f"Saved {OUT_PATH}")
    for name, t in [("X_train", X_train), ("y_train", y_train), ("X_test", X_test), ("y_test", y_test)]:
        print(f"  {name:8s} {str(t.dtype):12s} {tuple(t.shape)}")


if __name__ == "__main__":
    main()
