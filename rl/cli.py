from __future__ import annotations

import argparse
from pathlib import Path

from rl.common.config import Config
from rl.common.seeding import fold_in_seed
from rl.envs.brax_hexapod import BraxHexapodConfig
from rl.envs.mujoco_hexapod import HexapodEnvConfig
from rl.envs.terrain import HeightfieldSpec
from rl.training.ppo_brax import BraxPPOConfig, train_brax
from rl.training.ppo_torch import TorchPPOConfig, train_torch


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hexapod_static.xml"


def main() -> int:
    parser = argparse.ArgumentParser(description="Vendor-agnostic PPO trainer")
    parser.add_argument("--config", type=Path, default=Path("configs/ppo_baseline.json"))
    parser.add_argument("--backend", choices=["mujoco", "brax"], default=None)
    parser.add_argument("--terrain", choices=["flat", "heightfield"], default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num-seeds", type=int, default=None)
    args = parser.parse_args()

    cfg = Config.load(args.config)
    backend = args.backend or cfg.get("experiment", "backend")
    terrain = args.terrain or cfg.get("experiment", "terrain")
    base_seed = args.seed if args.seed is not None else cfg.get("experiment", "seed")
    num_seeds = args.num_seeds if args.num_seeds is not None else cfg.get("experiment", "num_seeds")

    env_cfg = cfg.data["env"]
    heightfield_cfg = env_cfg.get("heightfield", {})
    heightfield = HeightfieldSpec(
        size=tuple(heightfield_cfg.get("size", [4.0, 4.0])),
        field_res=tuple(heightfield_cfg.get("field_res", [32, 32])),
        height_scale=float(heightfield_cfg.get("height_scale", 0.05)),
        seed=int(heightfield_cfg.get("seed", 123)),
    )
    common_env_kwargs = dict(
        model_path=MODEL_PATH,
        episode_length=int(env_cfg["episode_length"]),
        action_repeat=int(env_cfg["action_repeat"]),
        target_velocity=float(env_cfg["target_velocity"]),
        ctrl_cost=float(env_cfg["ctrl_cost"]),
        orient_cost=float(env_cfg["orient_cost"]),
        joint_limit_cost=float(env_cfg["joint_limit_cost"]),
        alive_bonus=float(env_cfg["alive_bonus"]),
        reset_noise=float(env_cfg["reset_noise"]),
        terrain=terrain,
        heightfield=heightfield,
    )

    experiment = cfg.get("experiment", "name")
    log_root = Path(cfg.get("experiment", "log_dir")) / experiment / backend / terrain

    policy_cfg = cfg.data["policy"]
    eval_cfg = cfg.data.get("eval", {})
    checkpoint_cfg = cfg.data.get("checkpoint", {})
    eval_interval = int(eval_cfg.get("eval_interval", 200000))
    checkpoint_interval = int(checkpoint_cfg.get("save_interval", 500000))

    for offset in range(num_seeds):
        seed = fold_in_seed(base_seed, offset)
        log_dir = log_root / f"seed_{seed}"
        if backend == "mujoco":
            env = HexapodEnvConfig(**common_env_kwargs)
            ppo_cfg = TorchPPOConfig(**cfg.data["ppo"])
            train_torch(
                env,
                ppo_cfg,
                policy_cfg,
                log_dir,
                seed,
                cfg.get("checkpoint", "export_unified"),
                eval_interval,
                checkpoint_interval,
            )
        else:
            env = BraxHexapodConfig(**common_env_kwargs)
            ppo_cfg = BraxPPOConfig(**cfg.data["ppo"])
            train_brax(
                env,
                ppo_cfg,
                policy_cfg,
                log_dir,
                seed,
                cfg.get("checkpoint", "export_unified"),
                eval_interval,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
