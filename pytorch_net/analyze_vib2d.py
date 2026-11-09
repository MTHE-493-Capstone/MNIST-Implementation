import argparse
import json

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import torch

from analyze_vib import beta_label, finish, load_runs
from mnist_data import load_mnist
from model import VIB2D
from plot_style import SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, set_title, style_axes
from sweep import SWEEP_BETAS_2D
from train import CHECKPOINT_DIR, EVAL_SAMPLES, LOG_DIR, NUM_CLASSES, run_name

MODEL = "vib2d"
REPORT_TABLE_7 = dict(
    zip(
        SWEEP_BETAS_2D,
        [(3.98, 3.98), (3.87, 3.84), (3.62, 3.59), (3.33, 3.19), (3.12, 3.13), (4.86, 3.21), (27.25, 6.43), (84.59, 75.78)],
    )
)
EMBEDDING_BETAS = [0.0, 1e-4, 1e-3, 1e-1, 10**-0.5, 1.0]
EMBEDDING_SEED = 0
AXIS_QUANTILES = (0.005, 0.995)


def summarize(runs: dict[float, dict]) -> list[dict]:
    rows = []
    for beta, run in runs.items():
        final = run["final"]
        report_1, report_12 = REPORT_TABLE_7[beta]
        rows.append(
            {
                "beta": beta,
                "beta_label": beta_label(beta),
                "test_error_1": final["test_error_rate"],
                f"test_error_{EVAL_SAMPLES}": final[f"test_error_rate_{EVAL_SAMPLES}"],
                "report_test_error_1": report_1 / 100,
                f"report_test_error_{EVAL_SAMPLES}": report_12 / 100,
                "train_ixz_bits": final["train_ixz_bits"],
                "test_izy_bits": final["test_izy_bits"],
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    n = EVAL_SAMPLES
    print(f"{'':>8}  {'test error (ours)':^21}  {'test error (report)':^21}  {'I(X;Z)':>7}  {'I(Z;Y)':>7}")
    print(f"{'beta':>8}  {'1 sample':>10} {f'{n} samples':>10}  {'1 sample':>10} {f'{n} samples':>10}  {'train':>7}  {'test':>7}")
    for r in rows:
        print(
            f"{r['beta_label']:>8}  {r['test_error_1']:>10.2%} {r[f'test_error_{n}']:>10.2%}  "
            f"{r['report_test_error_1']:>10.2%} {r[f'report_test_error_{n}']:>10.2%}  "
            f"{r['train_ixz_bits']:>7.2f}  {r['test_izy_bits']:>7.3f}"
        )
    best = min(rows, key=lambda r: r[f"test_error_{n}"])
    print(f"\nbest {n}-sample test error: {best[f'test_error_{n}']:.2%} at beta = {best['beta_label']} (report: 3.12% / 3.13% at 10^-3)")


@torch.no_grad()
def embed_test_set(beta: float, seed: int, X_test: torch.Tensor) -> torch.Tensor:
    checkpoint = torch.load(CHECKPOINT_DIR / f"{run_name(MODEL, beta, seed)}.pt")
    model = VIB2D()
    model.load_state_dict(checkpoint["ema_model"])
    torch.manual_seed(EMBEDDING_SEED)
    return model.sample(*model.encode(X_test))


def plot_embeddings(rows: list[dict], seed: int) -> None:
    _, _, X_test, y_test = load_mnist("cpu")
    by_beta = {r["beta"]: r for r in rows}
    betas = [b for b in EMBEDDING_BETAS if b in by_beta]
    fig, axes = plt.subplots(3, 2, figsize=(11, 15), facecolor=SURFACE)
    for ax, beta in zip(axes.flat, betas):
        z = embed_test_set(beta, seed, X_test)
        ax.scatter(z[:, 0], z[:, 1], s=2, color=TEXT_SECONDARY, alpha=0.12, linewidths=0, rasterized=True)
        for digit in range(NUM_CLASSES):
            center = z[y_test == digit].median(dim=0).values
            ax.text(center[0], center[1], str(digit), fontsize=15, fontweight="bold", color=TEXT_PRIMARY,
                    ha="center", va="center",
                    path_effects=[path_effects.withStroke(linewidth=3, foreground=SURFACE)])
        low, high = torch.quantile(z, torch.tensor(AXIS_QUANTILES), dim=0)
        pad = 0.05 * (high - low)
        ax.set_xlim(low[0] - pad[0], high[0] + pad[0])
        ax.set_ylim(low[1] - pad[1], high[1] + pad[1])
        r = by_beta[beta]
        set_title(
            ax,
            f"β = {beta_label(beta, tex=True)}:  {r['test_error_1']:.2%} / {r[f'test_error_{EVAL_SAMPLES}']:.2%} error "
            f"(1 / {EVAL_SAMPLES} samples)",
        )
        ax.set_xlabel("z₁", color=TEXT_SECONDARY)
        ax.set_ylabel("z₂", color=TEXT_SECONDARY)
        style_axes(ax, None)
    for ax in list(axes.flat)[len(betas):]:
        ax.set_visible(False)
    finish(
        fig,
        "2D VIB embeddings of the 10,000 test images (one sampled z each; digit = class median; axes differ per panel)",
        "vib2d_embeddings.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    seed = parser.parse_args().seed
    runs = load_runs(MODEL, SWEEP_BETAS_2D, seed)
    rows = summarize(runs)
    print_table(rows)
    summary_path = LOG_DIR / f"{MODEL}_sweep_seed{seed}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"saved {summary_path}")
    plot_embeddings(rows, seed)


if __name__ == "__main__":
    main()
