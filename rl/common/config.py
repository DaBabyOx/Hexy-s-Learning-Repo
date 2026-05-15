from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict


DEFAULTS: Dict[str, Any] = {
    "experiment": {
        "name": "hexapod_ppo",
        "backend": "mujoco",
        "terrain": "flat",
        "num_seeds": 5,
        "seed": 0,
        "log_dir": "runs",
    },
    "env": {
        "episode_length": 1000,
        "action_repeat": 1,
        "target_velocity": 0.6,
        "ctrl_cost": 0.02,
        "orient_cost": 1.0,
        "joint_limit_cost": 0.2,
        "alive_bonus": 0.0,
        "reset_noise": 0.1,
        "heightfield": {
            "size": [4.0, 4.0],
            "field_res": [32, 32],
            "height_scale": 0.05,
            "seed": 123,
        },
    },
    "policy": {
        "hidden_sizes": [256, 256],
        "activation": "tanh",
        "log_std_init": -0.5,
    },
    "ppo": {
        "total_steps": 5_000_000,
        "num_envs": 256,
        "rollout_length": 128,
        "update_epochs": 4,
        "minibatch_size": 1024,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_epsilon": 0.2,
        "vf_coef": 0.5,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "normalize_obs": True,
    },
    "eval": {
        "eval_interval": 200_000,
        "num_eval_episodes": 8,
    },
    "checkpoint": {
        "save_interval": 500_000,
        "export_unified": True,
    },
}


@dataclass
class Config:
    data: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        merged = _merge_dicts(DEFAULTS, payload)
        return cls(merged)

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    def get(self, *keys: str, default: Any | None = None) -> Any:
        node: Any = self.data
        for key in keys:
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node


def _merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result
