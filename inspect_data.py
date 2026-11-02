from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "mnist.pt"
FIG_PATH = ROOT / "figures" / "first_16.png"


def print_summary(d: dict[str, torch.Tensor]) -> None:
    for name, t in d.items():
        print(f"{name:8s} {str(t.dtype):12s} {tuple(t.shape)}")

    for split in ("train", "test"):
        counts = torch.bincount(d[f"y_{split}"], minlength=10)
        print(f"\n{split} class counts (total {counts.sum().item()}):")
        for digit, count in enumerate(counts.tolist()):
            print(f"  {digit}: {count}")


def plot_first_16(X: torch.Tensor, y: torch.Tensor) -> None:
    fig, axes = plt.subplots(4, 4, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(X[i].reshape(28, 28), cmap="gray")
        ax.set_title(f"label: {y[i].item()}", fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=120)
    print(f"\nSaved {FIG_PATH}")
    plt.show()


def main() -> None:
    d = torch.load(DATA_PATH)
    print_summary(d)
    plot_first_16(d["X_train"], d["y_train"])


if __name__ == "__main__":
    main()
