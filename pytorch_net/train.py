import time

import torch
import torch.nn as nn

from losses import cross_entropy_bits
from mnist_data import batches, get_device, load_mnist
from model import MLP

LR = 1e-4
BETAS = (0.5, 0.999)
LR_STEP_EPOCHS = 2
LR_GAMMA = 0.97
EPOCHS = 200
BATCH_SIZE = 100
SEED = 0


def make_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)


def make_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    logits = model(X)
    return cross_entropy_bits(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()
    X_train, y_train, X_test, y_test = load_mnist(device)
    model = MLP().to(device)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    print(f"device {device}  epochs {EPOCHS}  batch {BATCH_SIZE}  steps/epoch {len(X_train) // BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]

        for xb, yb in batches(X_train, y_train, BATCH_SIZE):
            loss = cross_entropy_bits(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        train_loss, train_acc = evaluate(model, X_train, y_train)
        test_loss, test_acc = evaluate(model, X_test, y_test)
        print(
            f"epoch {epoch:3d}  lr {lr:.2e}  "
            f"train loss {train_loss:.4f} bits  acc {train_acc:.4f}  "
            f"test loss {test_loss:.4f} bits  acc {test_acc:.4f}  "
            f"({time.perf_counter() - start:.1f}s)"
        )


if __name__ == "__main__":
    main()
