import numpy as np

LAYER_SIZES = (784, 16, 16, 10)


def init_params(sizes: tuple[int, ...] = LAYER_SIZES, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    params = {}
    for layer, (n_in, n_out) in enumerate(zip(sizes[:-1], sizes[1:]), start=1):
        params[f"W{layer}"] = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        params[f"b{layer}"] = np.zeros(n_out)
    return params


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)


def softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def forward(params: dict[str, np.ndarray], X: np.ndarray) -> dict[str, np.ndarray]:
    z1 = X @ params["W1"] + params["b1"]
    a1 = relu(z1)

    z2 = a1 @ params["W2"] + params["b2"]
    a2 = relu(z2)

    z3 = a2 @ params["W3"] + params["b3"]
    a3 = softmax(z3)

    return {"a0": X, "z1": z1, "a1": a1, "z2": z2, "a2": a2, "z3": z3, "a3": a3}
