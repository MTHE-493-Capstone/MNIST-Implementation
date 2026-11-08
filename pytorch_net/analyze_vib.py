import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from plot_style import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, set_title, style_axes
from sweep import SWEEP_BETAS
from train import EVAL_SAMPLES, LOG_DIR, NUM_CLASSES, run_name

FIG_DIR = Path(__file__).parent.parent / "figures"
BASELINE_SUMMARY_PATH = LOG_DIR / "baseline_summary.json"
CURVE_BETAS = [0.0, 1e-3, 1.0]
PLANE_LABEL_BETAS = [0.0, 1e-5, 1e-3, 1e-1, 10**-0.5, 1.0]
ZERO_BETA_X = 10**-7
H_Y_BITS = math.log2(NUM_CLASSES)
REPORT_TABLE_5 = dict(
    zip(
        SWEEP_BETAS,
        [
            (1.47, 1.46), (1.40, 1.29), (1.46, 1.30), (1.37, 1.23), (1.43, 1.18), (1.50, 1.24), (1.64, 1.17),
            (1.75, 1.33), (2.10, 1.38), (1.55, 1.38), (3.91, 1.49), (9.14, 1.55), (24.20, 2.07), (85.48, 79.70),
        ],
    )
)


def beta_label(beta: float, tex: bool = False) -> str:
    if beta in (0, 1):
        return str(int(beta))
    exponent = f"{math.log10(beta):g}"
    return f"$10^{{{exponent}}}$" if tex else f"10^{exponent}"


def beta_x(beta: float) -> float:
    return beta if beta > 0 else ZERO_BETA_X


def load_runs(seed: int) -> dict[float, dict]:
    runs = {}
    for beta in SWEEP_BETAS:
        path = LOG_DIR / f"{run_name('vib', beta, seed)}.json"
        run = json.loads(path.read_text()) if path.exists() else {}
        if "final" in run:
            runs[beta] = run
        else:
            print(f"missing or unfinished: {path.name}")
    return runs


def training_objective_bits(run: dict) -> list[float]:
    beta = run["config"]["beta"]
    return [h["train_loss_bits"] + beta * h["train_ixz_bits"] for h in run["history"]]


def summarize(runs: dict[float, dict]) -> list[dict]:
    rows = []
    for beta, run in runs.items():
        final = run["final"]
        report_1, report_12 = REPORT_TABLE_5[beta]
        rows.append(
            {
                "beta": beta,
                "beta_label": beta_label(beta),
                "test_error_1": final["test_error_rate"],
                f"test_error_{EVAL_SAMPLES}": final[f"test_error_rate_{EVAL_SAMPLES}"],
                "report_test_error_1": report_1 / 100,
                f"report_test_error_{EVAL_SAMPLES}": report_12 / 100,
                "train_errors_1": final["train_errors"],
                f"train_errors_{EVAL_SAMPLES}": final[f"train_errors_{EVAL_SAMPLES}"],
                "train_ixz_bits": final["train_ixz_bits"],
                "test_ixz_bits": final["test_ixz_bits"],
                "train_izy_bits": final["train_izy_bits"],
                "test_izy_bits": final["test_izy_bits"],
            }
        )
    return rows


def print_table(rows: list[dict]) -> None:
    n = EVAL_SAMPLES
    print(f"{'':>8}  {'test error (ours)':^21}  {'test error (report)':^21}  {'train errors':^13}  {'I(X;Z)':>7}  {'I(Z;Y) bits':^15}")
    print(f"{'beta':>8}  {'1 sample':>10} {f'{n} samples':>10}  {'1 sample':>10} {f'{n} samples':>10}  {'1':>6} {n:>6}  {'train':>7}  {'train':>7} {'test':>7}")
    for r in rows:
        print(
            f"{r['beta_label']:>8}  {r['test_error_1']:>10.2%} {r[f'test_error_{n}']:>10.2%}  "
            f"{r['report_test_error_1']:>10.2%} {r[f'report_test_error_{n}']:>10.2%}  "
            f"{r['train_errors_1']:>6} {r[f'train_errors_{n}']:>6}  "
            f"{r['train_ixz_bits']:>7.1f}  {r['train_izy_bits']:>7.3f} {r['test_izy_bits']:>7.3f}"
        )
    best = min(rows, key=lambda r: r[f"test_error_{n}"])
    print(f"\nbest {n}-sample test error: {best[f'test_error_{n}']:.2%} at beta = {best['beta_label']} (report: 1.17% at 10^-3.5)")
    print(f"I(Z;Y) is at most H(Y) = log2({NUM_CLASSES}) = {H_Y_BITS:.3f} bits")


def print_training_check(runs: dict[float, dict]) -> None:
    for beta in CURVE_BETAS:
        if beta not in runs:
            continue
        run = runs[beta]
        objective = training_objective_bits(run)
        acc = [h[f"test_acc_{EVAL_SAMPLES}"] for h in run["history"]]
        print(
            f"beta {beta_label(beta):>6}: training loss (CE + beta*KL) {objective[0]:.3f} -> {objective[9]:.3f} -> {objective[-1]:.3f} bits"
            f" (epochs 1, 10, {len(objective)});  test acc ({EVAL_SAMPLES} samples) {acc[0]:.2%} -> {acc[9]:.2%} -> {acc[-1]:.2%}"
        )


def set_beta_axis(ax: plt.Axes, betas: list[float]) -> None:
    ax.set_xscale("log")
    decades = [b for b in betas if b > 0 and float(math.log10(b)).is_integer()]
    ticks = ([ZERO_BETA_X] if 0 in betas else []) + decades
    ax.set_xticks(ticks, [beta_label(b, tex=True) for b in ([0.0] if 0 in betas else []) + decades])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("β (β = 0 shown at the left)", color=TEXT_SECONDARY)


def plot_beta_series(ax: plt.Axes, betas: list[float], values: list[float], **style) -> None:
    pairs = [(beta_x(b), v) for b, v in zip(betas, values) if b > 0]
    ax.plot(*zip(*pairs), **style)
    if 0 in betas:
        zero_style = {k: v for k, v in style.items() if k != "label"}
        ax.plot([ZERO_BETA_X], [values[betas.index(0)]], **zero_style)


def finish(fig: plt.Figure, title: str, name: str) -> None:
    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    FIG_DIR.mkdir(exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {path}")


def plot_training_curves(runs: dict[float, dict]) -> None:
    fig, (ax_loss, ax_err) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, facecolor=SURFACE)
    for color, beta in zip(SERIES, CURVE_BETAS):
        if beta not in runs:
            continue
        run = runs[beta]
        epochs = [h["epoch"] for h in run["history"]]
        error = [100 * (1 - h[f"test_acc_{EVAL_SAMPLES}"]) for h in run["history"]]
        label = f"β = {beta_label(beta, tex=True)}"
        ax_loss.plot(epochs, training_objective_bits(run), color=color, linewidth=1.5, label=label)
        ax_err.plot(epochs, error, color=color, linewidth=1.5, label=f"{label} (final {error[-1]:.2f}%)")
    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("CE + β·KL (bits, log scale)", color=TEXT_SECONDARY)
    set_title(ax_loss, "Training loss (EMA model, full training set)")
    ax_err.set_yscale("log")
    ax_err.set_ylabel("test error (%, log scale)", color=TEXT_SECONDARY)
    ax_err.set_xlabel("epoch", color=TEXT_SECONDARY)
    set_title(ax_err, f"Test error ({EVAL_SAMPLES}-sample average)")
    style_axes(ax_loss, "upper right")
    style_axes(ax_err, "center right")
    finish(fig, "VIB training: β = 0, 10⁻³ and 1", "vib_training_curves.png")


def plot_error_vs_beta(rows: list[dict]) -> None:
    n = EVAL_SAMPLES
    betas = [r["beta"] for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.5), facecolor=SURFACE)
    for color, key, name in ((SERIES[0], "1", "1 sample"), (SERIES[1], f"{n}", f"{n}-sample average")):
        ours = [100 * r[f"test_error_{key}"] for r in rows]
        report = [100 * r[f"report_test_error_{key}"] for r in rows]
        plot_beta_series(ax, betas, ours, color=color, linewidth=2, marker="o", markersize=7,
                         markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"ours, {name}", zorder=4)
        plot_beta_series(ax, betas, report, color=color, linewidth=1.2, linestyle=(0, (4, 3)), marker="o",
                         markersize=7, markerfacecolor=SURFACE, markeredgewidth=1.5, label=f"report, {name}", zorder=3)
    if BASELINE_SUMMARY_PATH.exists():
        baseline = 100 * json.loads(BASELINE_SUMMARY_PATH.read_text())["final_test_error_mean"]
        ax.axhline(baseline, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (1, 2)),
                   label=f"deterministic baseline, ours ({baseline:.2f}%)")
    set_beta_axis(ax, betas)
    ax.set_yscale("log")
    ax.set_yticks([1, 2, 5, 10, 20, 50, 100], ["1", "2", "5", "10", "20", "50", "100"])
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_ylabel("test error (%, log scale)", color=TEXT_SECONDARY)
    set_title(ax, "Test error after 200 epochs (reproduces Table 5 / Figure 51)")
    style_axes(ax, "upper left")
    finish(fig, "VIB test error vs β", "vib_error_vs_beta.png")


def plot_information(rows: list[dict]) -> None:
    betas = [r["beta"] for r in rows]
    fig, (ax_ixz, ax_plane) = plt.subplots(1, 2, figsize=(12, 5.5), facecolor=SURFACE)

    plot_beta_series(ax_ixz, betas, [r["train_ixz_bits"] for r in rows], color=SERIES[0], linewidth=2,
                     marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label="train")
    set_beta_axis(ax_ixz, betas)
    ax_ixz.set_yscale("log")
    ax_ixz.set_ylabel("Î(X;Z) upper bound (bits, log scale)", color=TEXT_SECONDARY)
    set_title(ax_ixz, "Compression: larger β keeps fewer bits (Fig. 52)")
    style_axes(ax_ixz, None)

    for color, split in ((SERIES[0], "train"), (SERIES[1], "test")):
        ax_plane.plot([r[f"{split}_ixz_bits"] for r in rows], [r[f"{split}_izy_bits"] for r in rows], color=color,
                      linewidth=1.5, marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.5, label=split)
    for r in rows:
        if r["beta"] in PLANE_LABEL_BETAS:
            above = r["train_izy_bits"] > 3.2
            ax_plane.annotate(f"β={beta_label(r['beta'], tex=True)}", (r["train_ixz_bits"], r["train_izy_bits"]),
                              textcoords="offset points", xytext=(-12, 8) if above else (8, -4),
                              fontsize=8, color=TEXT_SECONDARY)
    ax_plane.axhline(H_Y_BITS, color=TEXT_SECONDARY, linewidth=1, linestyle=(0, (1, 2)),
                     label=f"H(Y) = log₂{NUM_CLASSES} = {H_Y_BITS:.2f} bits")
    ax_plane.set_xscale("log")
    ax_plane.set_xlim(right=3 * max(r["train_ixz_bits"] for r in rows))
    ax_plane.set_xlabel("Î(X;Z) upper bound (bits, log scale)", color=TEXT_SECONDARY)
    ax_plane.set_ylabel("Î(Z;Y) lower bound (bits)", color=TEXT_SECONDARY)
    set_title(ax_plane, "Information plane (Fig. 53)")
    style_axes(ax_plane, "lower right")
    finish(fig, "VIB information bounds after 200 epochs", "vib_information.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    seed = parser.parse_args().seed
    runs = load_runs(seed)
    rows = summarize(runs)
    print_table(rows)
    print()
    print_training_check(runs)
    summary_path = LOG_DIR / f"vib_sweep_seed{seed}_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2))
    print(f"saved {summary_path}")
    plot_training_curves(runs)
    plot_error_vs_beta(rows)
    plot_information(rows)


if __name__ == "__main__":
    main()
