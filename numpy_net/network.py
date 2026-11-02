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


def one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    return np.eye(num_classes)[y]


def relu_grad(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def cross_entropy(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(probs[np.arange(len(y)), y]).mean())


def backward(params: dict[str, np.ndarray], cache: dict[str, np.ndarray], y: np.ndarray) -> dict[str, np.ndarray]:
    batch_size = len(y)

    delta3 = (cache["a3"] - one_hot(y)) / batch_size
    dW3 = cache["a2"].T @ delta3
    db3 = delta3.sum(axis=0)

    delta2 = (delta3 @ params["W3"].T) * relu_grad(cache["z2"])
    dW2 = cache["a1"].T @ delta2
    db2 = delta2.sum(axis=0)

    delta1 = (delta2 @ params["W2"].T) * relu_grad(cache["z1"])
    dW1 = cache["a0"].T @ delta1
    db1 = delta1.sum(axis=0)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
