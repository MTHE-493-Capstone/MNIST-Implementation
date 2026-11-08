import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import LN2, cross_entropy_bits, gaussian_kl, vib_loss_bits
from mnist_data import batches, get_device, load_mnist
from model import MLP, VIB, Z_DIM

LR = 1e-4
BETAS = (0.5, 0.999)
LR_STEP_EPOCHS = 2
LR_GAMMA = 0.97
EPOCHS = 200
BATCH_SIZE = 100
EMA_DECAY = 0.999
DEFAULT_BETA = 10**-3.5
EVAL_SAMPLES = 12
NUM_CLASSES = 10
LOG_DIR = Path(__file__).parent.parent / "logs"
CHECKPOINT_DIR = Path(__file__).parent.parent / "checkpoints"


def run_name(model: str, beta: float, seed: int) -> str:
    return f"vib_beta{beta:.2e}_seed{seed}" if model == "vib" else f"baseline_seed{seed}"


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


def training_loss(model: nn.Module, xb: torch.Tensor, yb: torch.Tensor, beta: float) -> torch.Tensor:
    if isinstance(model, VIB):
        logits, mu, sigma = model(xb)
        return vib_loss_bits(logits, yb, mu, sigma, beta)[0]
    return cross_entropy_bits(model(xb), yb)


def score(log_probs: torch.Tensor, y: torch.Tensor, suffix: str = "") -> dict[str, float | int]:
    errors = int((log_probs.argmax(dim=1) != y).sum().item())
    return {
        f"loss_bits{suffix}": (F.nll_loss(log_probs, y) / LN2).item(),
        f"acc{suffix}": 1 - errors / len(y),
        f"errors{suffix}": errors,
    }


@torch.no_grad()
def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> dict[str, float | int]:
    if not isinstance(model, VIB):
        return score(F.log_softmax(model(X), dim=1), y)
    mu, sigma = model.encode(X)
    log_probs = torch.stack(
        [F.log_softmax(model.decoder(model.sample(mu, sigma)), dim=1) for _ in range(EVAL_SAMPLES)]
    )
    mean_log_probs = torch.logsumexp(log_probs, dim=0) - math.log(EVAL_SAMPLES)
    one_sample = score(log_probs[0], y)
    return {
        **one_sample,
        **score(mean_log_probs, y, f"_{EVAL_SAMPLES}"),
        "ixz_bits": (gaussian_kl(mu, sigma) / LN2).item(),
        "izy_bits": math.log2(NUM_CLASSES) - one_sample["loss_bits"],
    }


def describe(metrics: dict[str, float | int]) -> str:
    text = f"loss {metrics['loss_bits']:.4f} bits  acc {metrics['acc']:.4f}"
    if "ixz_bits" in metrics:
        text += (
            f"  acc{EVAL_SAMPLES} {metrics[f'acc_{EVAL_SAMPLES}']:.4f}"
            f"  I(X;Z) {metrics['ixz_bits']:.1f}  I(Z;Y) {metrics['izy_bits']:.3f} bits"
        )
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["baseline", "vib"], default="baseline")
    parser.add_argument("--beta", type=float, default=DEFAULT_BETA)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    is_vib = args.model == "vib"
    name = run_name(args.model, args.beta, args.seed)
    log_path = LOG_DIR / f"{name}.json"
    checkpoint_path = CHECKPOINT_DIR / f"{name}.pt"

    torch.manual_seed(args.seed)
    device = get_device()
    X_train, y_train, X_test, y_test = load_mnist(device)
    model = (VIB() if is_vib else MLP()).to(device)
    ema_model = make_ema(model)
    optimizer = make_optimizer(model)
    scheduler = make_scheduler(optimizer)
    config = {
        "model": args.model,
        "lr": LR,
        "betas": BETAS,
        "lr_step_epochs": LR_STEP_EPOCHS,
        "lr_gamma": LR_GAMMA,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": args.seed,
        "ema_decay": EMA_DECAY,
    }
    if is_vib:
        config |= {"beta": args.beta, "z_dim": Z_DIM, "eval_samples": EVAL_SAMPLES}
    history = []
    LOG_DIR.mkdir(exist_ok=True)
    print(f"{name}  device {device}  epochs {EPOCHS}  batch {BATCH_SIZE}  steps/epoch {len(X_train) // BATCH_SIZE}")

    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]

        for xb, yb in batches(X_train, y_train, BATCH_SIZE):
            loss = training_loss(model, xb, yb, args.beta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema(ema_model, model)
        scheduler.step()

        train = evaluate(ema_model, X_train, y_train)
        test = evaluate(ema_model, X_test, y_test)
        raw_test = evaluate(model, X_test, y_test)
        print(
            f"epoch {epoch:3d}  lr {lr:.2e}  train {describe(train)}  test {describe(test)}  "
            f"(raw model test acc {raw_test['acc']:.4f})  ({time.perf_counter() - start:.1f}s)"
        )

        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                **{f"train_{k}": v for k, v in train.items()},
                **{f"test_{k}": v for k, v in test.items()},
            }
        )
        log_path.write_text(json.dumps({"config": config, "history": history}, indent=2))

    final = {
        "test_errors": test["errors"],
        "test_error_rate": test["errors"] / len(y_test),
        "train_errors": train["errors"],
        "raw_model_train_errors": evaluate(model, X_train, y_train)["errors"],
    }
    if is_vib:
        final |= {
            f"test_errors_{EVAL_SAMPLES}": test[f"errors_{EVAL_SAMPLES}"],
            f"test_error_rate_{EVAL_SAMPLES}": test[f"errors_{EVAL_SAMPLES}"] / len(y_test),
            f"train_errors_{EVAL_SAMPLES}": train[f"errors_{EVAL_SAMPLES}"],
            "train_ixz_bits": train["ixz_bits"],
            "train_izy_bits": train["izy_bits"],
            "test_ixz_bits": test["ixz_bits"],
            "test_izy_bits": test["izy_bits"],
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
