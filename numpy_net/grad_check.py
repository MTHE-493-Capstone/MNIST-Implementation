import numpy as np

from dataset import load_mnist
from network import backward, cross_entropy, forward, init_params

TOLERANCE = 1e-6


def gradient_check(
    params: dict[str, np.ndarray],
    X: np.ndarray,
    y: np.ndarray,
    num_checks: int = 20,
    eps: float = 1e-5,
    seed: int = 0,
) -> float:
    assert X.dtype == np.float64 and all(p.dtype == np.float64 for p in params.values())
    rng = np.random.default_rng(seed)
    grads = backward(params, forward(params, X), y)
    names = list(params)
    worst = 0.0

    for _ in range(num_checks):
        name = names[rng.integers(len(names))]
        idx = tuple(int(rng.integers(n)) for n in params[name].shape)

        original = params[name][idx]
        params[name][idx] = original + eps
        loss_plus = cross_entropy(forward(params, X)["a3"], y)
        params[name][idx] = original - eps
        loss_minus = cross_entropy(forward(params, X)["a3"], y)
        params[name][idx] = original

        numeric = (loss_plus - loss_minus) / (2 * eps)
        analytic = grads[name][idx]
        rel_error = abs(analytic - numeric) / max(abs(analytic) + abs(numeric), 1e-12)
        worst = max(worst, rel_error)
        print(f"{name}{list(idx)!s:12s} analytic {analytic:+.6e}  numeric {numeric:+.6e}  rel error {rel_error:.2e}")

    return worst


def main() -> None:
    X_train, y_train, _, _ = load_mnist()
    batch = np.random.default_rng(0).choice(len(X_train), size=100, replace=False)
    worst = gradient_check(init_params(), X_train[batch], y_train[batch])
    print(f"\nworst relative error {worst:.2e} (tolerance {TOLERANCE:.0e})")
    assert worst < TOLERANCE, "gradient check failed: do not train"
    print("PASSED")


if __name__ == "__main__":
    main()
