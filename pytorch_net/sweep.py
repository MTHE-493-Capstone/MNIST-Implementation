import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from train import LOG_DIR, run_name

SWEEP_LOG10_BETAS = [-6 + 0.5 * k for k in range(13)]
SWEEP_BETAS = [0.0] + [10**b for b in SWEEP_LOG10_BETAS]
SWEEP_BETAS_2D = [0.0, 1e-5, 1e-4, 10**-3.5, 1e-3, 1e-1, 10**-0.5, 1.0]
MODEL_BETAS = {"vib": SWEEP_BETAS, "vib2d": SWEEP_BETAS_2D}
TRAIN_SCRIPT = Path(__file__).parent / "train.py"


def run(model: str, beta: float, seed: int) -> None:
    name = run_name(model, beta, seed)
    with open(LOG_DIR / f"{name}.out", "w") as out:
        subprocess.run(
            [sys.executable, TRAIN_SCRIPT, "--model", model, "--beta", repr(beta), "--seed", str(seed)],
            stdout=out,
            stderr=subprocess.STDOUT,
            check=True,
        )
    print(f"done {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODEL_BETAS), default="vib")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    LOG_DIR.mkdir(exist_ok=True)
    with ThreadPoolExecutor(args.workers) as pool:
        list(pool.map(lambda beta: run(args.model, beta, args.seed), MODEL_BETAS[args.model]))


if __name__ == "__main__":
    main()
