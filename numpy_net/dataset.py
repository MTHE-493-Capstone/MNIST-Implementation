from pathlib import Path

import numpy as np
import torch

DATA_PATH = Path(__file__).parent.parent / "data" / "mnist.pt"


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    d = torch.load(DATA_PATH)
    X_train = d["X_train"].numpy().astype(np.float64) / 255
    X_test = d["X_test"].numpy().astype(np.float64) / 255
    return X_train, d["y_train"].numpy(), X_test, d["y_test"].numpy()
