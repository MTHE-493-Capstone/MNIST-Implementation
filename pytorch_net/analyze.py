import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt

from plot_style import SERIES, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, set_title, style_axes

ROOT = Path(__file__).parent.parent
LOG_DIR = ROOT / "logs"
FIG_PATH = ROOT / "figures" / "baseline_curves.png"
SUMMARY_PATH = LOG_DIR / "baseline_summary.json"
REPORT_TRAIN_ERRORS = 28

TEST_ACC_FLOOR = 97.0
TEST_ACC_CEILING = 99.0


def load_runs() -> list[dict]:
    runs = [json.loads(p.read_text()) for p in sorted(LOG_DIR.glob("baseline_seed*.json"))]
    return sorted(runs, key=lambda r: r["config"]["seed"])


def best_test_epoch(run: dict) -> dict:
    return max(run["history"], key=lambda h: h["test_acc"])


def summarize(runs: list[dict]) -> dict:
    rows = []
    for run in runs:
        best = best_test_epoch(run)
        rows.append(
            {
                "seed": run["config"]["seed"],
                "final_test_errors": run["final"]["test_errors"],
                "final_test_error_rate": run["final"]["test_error_rate"],
                "final_train_errors": run["final"]["train_errors"],
                "final_raw_model_train_errors": run["final"]["raw_model_train_errors"],
                "best_test_epoch_analysis_only": best["epoch"],
                "best_test_acc_analysis_only": best["test_acc"],
            }
        )
    rates = [r["final_test_error_rate"] for r in rows]
    return {
        "runs": rows,
        "final_test_error_mean": statistics.mean(rates),
        "final_test_error_std": statistics.stdev(rates),
        "std_type": "sample (n - 1)",
        "report_train_errors": REPORT_TRAIN_ERRORS,
    }


def print_summary(summary: dict) -> None:
    print(f"{'seed':>4}  {'test errors':>11}  {'test error':>10}  {'train errors (EMA / raw)':>24}  {'best test epoch*':>16}")
    for r in summary["runs"]:
        print(
            f"{r['seed']:>4}  {r['final_test_errors']:>11}  {r['final_test_error_rate']:>9.2%}  "
            f"{r['final_train_errors']:>11} / {r['final_raw_model_train_errors']:<11}  "
            f"{r['best_test_epoch_analysis_only']:>6} ({r['best_test_acc_analysis_only']:.2%})"
        )
    print(
        f"\nfinal test error: mean {summary['final_test_error_mean']:.2%}  "
        f"std {summary['final_test_error_std']:.2%} ({summary['std_type']})"
    )
    print(f"report: {REPORT_TRAIN_ERRORS} / 60000 training errors")
    print("* best test epoch is for analysis only; reported results use the final epoch (200)")


def plot(runs: list[dict]) -> None:
    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(9, 7.5), sharex=True, facecolor=SURFACE)
    last_epoch = runs[0]["history"][-1]["epoch"]

    for color, run in zip(SERIES, runs):
        seed = run["config"]["seed"]
        epochs = [h["epoch"] for h in run["history"]]
        train_loss = [h["train_loss_bits"] for h in run["history"]]
        test_acc = [100 * h["test_acc"] for h in run["history"]]
        best = best_test_epoch(run)

        ax_loss.plot(epochs, train_loss, color=color, linewidth=1.5, label=f"seed {seed}")
        ax_acc.plot(epochs, test_acc, color=color, linewidth=1.5, label=f"seed {seed} (final {test_acc[-1]:.2f}%)")
        ax_acc.axvline(best["epoch"], color=color, linewidth=1, linestyle=(0, (3, 3)), alpha=0.7)
        ax_acc.plot(
            best["epoch"], 100 * best["test_acc"], marker="o", markersize=7,
            color=color, markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=5,
        )

    ax_acc.plot([], [], marker="o", linestyle=(0, (3, 3)), color=TEXT_SECONDARY,
                markeredgecolor=SURFACE, label="best test epoch (analysis only)")

    ax_loss.set_yscale("log")
    ax_loss.set_ylabel("training loss (bits, log scale)", color=TEXT_SECONDARY)
    set_title(ax_loss, "Training loss (EMA model, full training set)")
    ax_acc.set_ylim(TEST_ACC_FLOOR, TEST_ACC_CEILING)
    ax_acc.set_ylabel(f"test accuracy (%), axis from {TEST_ACC_FLOOR:.0f}%", color=TEXT_SECONDARY)
    ax_acc.set_xlabel("epoch", color=TEXT_SECONDARY)
    ax_acc.set_xlim(0, last_epoch + 2)
    set_title(ax_acc, "Test accuracy (EMA model); results reported at epoch 200")

    style_axes(ax_loss, "upper right")
    style_axes(ax_acc, "lower right")

    fig.suptitle("Deterministic baseline, 784-1024-1024-10, 3 seeds", x=0.01, ha="left", color=TEXT_PRIMARY, fontsize=13)
    fig.tight_layout()
    FIG_PATH.parent.mkdir(exist_ok=True)
    fig.savefig(FIG_PATH, dpi=150, facecolor=SURFACE)
    print(f"saved {FIG_PATH}")


def main() -> None:
    runs = load_runs()
    summary = summarize(runs)
    print_summary(summary)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"saved {SUMMARY_PATH}")
    plot(runs)


if __name__ == "__main__":
    main()
