import argparse
import copy
import json
import time
from pathlib import Path

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
EMA_DECAY = 0.999
LOG_DIR = Path(__file__).parent.parent / "logs"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def make_optimizer(model: nn.Module) -> torch.optim.Adam:
    return torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS)


def make_scheduler(optimizer: torch.optim.Optimizer) -> torch.optim.lr_scheduler.StepLR:
    return torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_EPOCHS, gamma=LR_GAMMA)


def make_ema(model: nn.Module) -> nn.Module:
    ema_model = copy.deepcopy(model)
    ema_model.requires_grad_(False)
    return ema_model


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module) -> None:
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(EMA_DECAY).add_(p, alpha=1 - EMA_DECAY)


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {name: t.cpu() for name, t in model.state_dict().items()}


@torch.no_grad()
def count_errors(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> int:
    return int((model(X).argmax(dim=1) != y).sum().item())


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> tuple[float, float]:
    logits = model(X)
    return cross_entropy_bits(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    seed = parser.parse_args().seed
    log_path = LOG_DIR / f"baseline_seed{seed}.json"
    checkpoint_path = CHECKPOINT_DIR / f"baseline_seed{seed}.pt"

    torch.manual_seed(seed)
    device = get_device()
    X_train, y_train, X_test, y_test = load_mnist(device)
    model = MLP().to(device)
    ema_model = make_ema(model)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    config = {
        "lr": LR,
        "betas": BETAS,
        "lr_step_epochs": LR_STEP_EPOCHS,
        "lr_gamma": LR_GAMMA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": seed,
        "ema_decay": EMA_DECAY,
    }
    history = []
    LOG_DIR.mkdir(exist_ok=True)
    print(f"seed {seed}  device {device}  epochs {EPOCHS}  batch {BATCH_SIZE}  steps/epoch {len(X_train) // BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]

        for xb, yb in batches(X_train, y_train, BATCH_SIZE):
            loss = cross_entropy_bits(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model)
        scheduler.step()

        train_loss, train_acc = evaluate(ema_model, X_train, y_train)
        test_loss, test_acc = evaluate(ema_model, X_test, y_test)
        _, raw_test_acc = evaluate(model, X_test, y_test)
        print(
            f"epoch {epoch:3d}  lr {lr:.2e}  "
            f"train loss {train_loss:.4f} bits  acc {train_acc:.4f}  "
            f"test loss {test_loss:.4f} bits  acc {test_acc:.4f}  "
            f"(raw model test acc {raw_test_acc:.4f})  "
            f"({time.perf_counter() - start:.1f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss_bits": train_loss,
                "train_acc": train_acc,
                "test_loss_bits": test_loss,
                "test_acc": test_acc,
            }
        )
        log_path.write_text(json.dumps({"config": config, "history": history}, indent=2))

    test_errors = count_errors(ema_model, X_test, y_test)
    final = {
        "test_errors": test_errors,
        "test_error_rate": test_errors / len(y_test),
        "train_errors": count_errors(ema_model, X_train, y_train),
        "raw_model_train_errors": count_errors(model, X_train, y_train),
    }
    print(f"final: {final}")
    log_path.write_text(json.dumps({"config": config, "final": final, "history": history}, indent=2))

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    torch.save(
        {
            "config": config,
            "final": final,
            "history": history,
            "model": cpu_state_dict(model),
            "ema_model": cpu_state_dict(ema_model),
        },
        checkpoint_path,
    )
    print(f"saved {log_path} and {checkpoint_path}")


if __name__ == "__main__":
    main()
