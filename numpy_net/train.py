import time

import numpy as np

from dataset import load_mnist
from grad_check import TOLERANCE, gradient_check
from network import backward, cross_entropy, forward, init_params

EPOCHS = 10
BATCH_SIZE = 100
LR = 0.1
SEED = 0


def accuracy(params: dict[str, np.ndarray], X: np.ndarray, y: np.ndarray) -> float:
    return float((forward(params, X)["a3"].argmax(axis=1) == y).mean())


def main() -> None:
    X_train, y_train, X_test, y_test = load_mnist()
    rng = np.random.default_rng(SEED)
    params = init_params(seed=SEED)

    check_batch = rng.choice(len(X_train), size=BATCH_SIZE, replace=False)
    worst = gradient_check(params, X_train[check_batch], y_train[check_batch])
    assert worst < TOLERANCE, f"gradient check failed ({worst:.2e}): not training"
    print(f"gradient check passed ({worst:.2e})\n")

    steps_per_epoch = len(X_train) // BATCH_SIZE
    for epoch in range(1, EPOCHS + 1):
        start = time.perf_counter()
        perm = rng.permutation(len(X_train))
        total_loss = 0.0

        for i in range(0, len(X_train), BATCH_SIZE):
            idx = perm[i : i + BATCH_SIZE]
            cache = forward(params, X_train[idx])
            total_loss += cross_entropy(cache["a3"], y_train[idx])
            grads = backward(params, cache, y_train[idx])
            for name in params:
                params[name] -= LR * grads[name]

        print(
            f"epoch {epoch:2d}  loss {total_loss / steps_per_epoch:.4f}  "
            f"train acc {accuracy(params, X_train, y_train):.4f}  "
            f"test acc {accuracy(params, X_test, y_test):.4f}  "
            f"({time.perf_counter() - start:.1f}s)"
        )


if __name__ == "__main__":
    main()
