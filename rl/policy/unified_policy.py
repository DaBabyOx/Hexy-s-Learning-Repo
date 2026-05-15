from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np


@dataclass
class UnifiedPolicy:
    weights: List[np.ndarray]
    biases: List[np.ndarray]
    mean: np.ndarray
    std: np.ndarray
    activation: str

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        x = (obs - self.mean) / (self.std + 1e-8)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            x = _activate(x @ w + b, self.activation)
        return np.tanh(x @ self.weights[-1] + self.biases[-1])

    @staticmethod
    def load(path: str | Path) -> "UnifiedPolicy":
        data = np.load(path, allow_pickle=True)
        weights = [data[f"w{i}"] for i in range(int(data["num_layers"]))]
        biases = [data[f"b{i}"] for i in range(int(data["num_layers"]))]
        mean = data["obs_mean"]
        std = data["obs_std"]
        activation = str(data["activation"])
        return UnifiedPolicy(weights, biases, mean, std, activation)


def export_unified_policy(
    weights: List[np.ndarray],
    biases: List[np.ndarray],
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    activation: str,
    path: str | Path,
) -> None:
    path = Path(path)
    payload = {
        "num_layers": len(weights),
        "obs_mean": obs_mean,
        "obs_std": obs_std,
        "activation": activation,
    }
    for idx, (w, b) in enumerate(zip(weights, biases)):
        payload[f"w{idx}"] = w
        payload[f"b{idx}"] = b
    np.savez(path, **payload)


def extract_torch_mlp(model) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for layer in model._module.layers:
        weights.append(layer.weight.detach().cpu().numpy().T)
        biases.append(layer.bias.detach().cpu().numpy())
    weights.append(model._module.policy_mean.weight.detach().cpu().numpy().T)
    biases.append(model._module.policy_mean.bias.detach().cpu().numpy())
    return weights, biases


def extract_brax_mlp(params) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    from flax.traverse_util import flatten_dict

    state = flatten_dict(params, sep="/")
    dense = {}
    for key, value in state.items():
        if not key.endswith("kernel") and not key.endswith("bias"):
            continue
        parts = key.split("/")
        dense_name = next((p for p in parts if p.startswith("Dense_")), None)
        if dense_name is None:
            continue
        idx = int(dense_name.split("_")[1])
        dense.setdefault(idx, {})["kernel" if key.endswith("kernel") else "bias"] = np.array(value)

    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
    for idx in sorted(dense.keys()):
        weights.append(np.array(dense[idx]["kernel"]))
        biases.append(np.array(dense[idx]["bias"]))
    return weights, biases


def _activate(x: np.ndarray, activation: str) -> np.ndarray:
    if activation == "tanh":
        return np.tanh(x)
    if activation == "relu":
        return np.maximum(0.0, x)
    raise ValueError(f"Unsupported activation: {activation}")
