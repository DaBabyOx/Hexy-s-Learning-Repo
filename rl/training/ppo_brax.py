from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import jax
import numpy as np

# Brax calls jax.device_put_replicated which was removed in newer JAX.
# Patch it back before importing Brax so its internal calls succeed.
# Brax's _unpmap does x.addressable_shards[0].data.squeeze(0), so each shard
# must have a leading size-1 axis — achieved by sharding a (n_devices, ...) array
# along the replica axis so each device owns exactly one slice of shape (1, ...).
try:
    if not callable(jax.device_put_replicated):
        raise AttributeError
except AttributeError:
    import numpy as _np
    import jax.numpy as _jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

    def _device_put_replicated(x, devices):
        n = len(devices)
        mesh = Mesh(_np.array(devices), ('replica',))
        sharding = NamedSharding(mesh, P('replica'))

        def _replicate(leaf):
            leaf_rep = _jnp.broadcast_to(
                _jnp.expand_dims(leaf, 0), (n,) + _jnp.shape(leaf)
            )
            return jax.device_put(leaf_rep, sharding)

        return jax.tree_util.tree_map(_replicate, x)

    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train

from rl.common.logger import CsvLogger
from rl.common.seeding import seed_all
from rl.envs.brax_hexapod import BraxHexapodConfig, BraxHexapodEnv
from rl.envs.terrain import HeightfieldSpec
from rl.policy.unified_policy import export_unified_policy, extract_brax_mlp


@dataclass
class BraxPPOConfig:
    total_steps: int
    num_envs: int
    rollout_length: int
    update_epochs: int
    minibatch_size: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_epsilon: float
    vf_coef: float
    ent_coef: float
    max_grad_norm: float
    normalize_obs: bool


def train_brax(
    env_cfg: BraxHexapodConfig,
    ppo_cfg: BraxPPOConfig,
    policy_cfg: Dict[str, float | List[int] | str],
    log_dir: Path,
    seed: int,
    export_unified: bool,
    eval_interval: int,
) -> None:
    seed_all(seed)
    env = BraxHexapodEnv(env_cfg, backend="mjx")
    logger = CsvLogger(log_dir)
    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logger.log_config({
        "backend": "brax",
        "seed": seed,
        "policy": policy_cfg,
        "ppo": ppo_cfg.__dict__,
        "env": env_cfg.__dict__,
    })

    policy_hidden = tuple(int(v) for v in policy_cfg["hidden_sizes"])
    network_factory = lambda obs_size, action_size, preprocess_observations_fn: ppo_networks.make_ppo_networks(
        obs_size,
        action_size,
        policy_hidden_layer_sizes=policy_hidden,
        value_hidden_layer_sizes=policy_hidden,
        activation=jax.nn.tanh if policy_cfg["activation"] == "tanh" else jax.nn.relu,
        preprocess_observations_fn=preprocess_observations_fn,
    )

    def progress_fn(step, metrics):
        log_metrics = {}
        for k, v in metrics.items():
            try:
                log_metrics[k] = float(v)
            except (TypeError, ValueError):
                pass
        logger.log(step, log_metrics)

    def policy_params_fn(step, make_policy, params):
        if not export_unified:
            return
        normalizer, policy_params, _ = params
        obs_mean = np.array(normalizer.mean)
        obs_std = np.array(normalizer.std)
        weights, biases = extract_brax_mlp(policy_params)
        export_unified_policy(
            weights,
            biases,
            obs_mean,
            obs_std,
            str(policy_cfg["activation"]),
            ckpt_dir / f"policy_unified_{step}.npz",
        )

    num_evals = max(2, int(ppo_cfg.total_steps // max(1, eval_interval)))
    n_devices = jax.local_device_count()
    if n_devices > 1:
        import warnings
        warnings.warn(
            f"JAX sees {n_devices} logical devices (MI300X multi-GCD). "
            "Restricting Brax pmap to 1 device via max_devices_per_host=1."
        )
    (make_policy, params, metrics) = ppo_train(
        env,
        num_timesteps=ppo_cfg.total_steps,
        episode_length=env_cfg.episode_length,
        action_repeat=env_cfg.action_repeat,
        num_envs=ppo_cfg.num_envs,
        max_devices_per_host=1,
        unroll_length=ppo_cfg.rollout_length,
        batch_size=ppo_cfg.minibatch_size,
        num_minibatches=max(1, (ppo_cfg.num_envs * ppo_cfg.rollout_length) // ppo_cfg.minibatch_size),
        num_updates_per_batch=ppo_cfg.update_epochs,
        learning_rate=ppo_cfg.learning_rate,
        discounting=ppo_cfg.gamma,
        gae_lambda=ppo_cfg.gae_lambda,
        clipping_epsilon=ppo_cfg.clip_epsilon,
        vf_loss_coefficient=ppo_cfg.vf_coef,
        entropy_cost=ppo_cfg.ent_coef,
        max_grad_norm=ppo_cfg.max_grad_norm,
        normalize_observations=ppo_cfg.normalize_obs,
        seed=seed,
        num_evals=num_evals,
        num_eval_envs=128,
        progress_fn=progress_fn,
        policy_params_fn=policy_params_fn,
        wrap_env=True,
        log_training_metrics=True,
    )

    if export_unified:
        normalizer, policy_params, _ = params
        weights, biases = extract_brax_mlp(policy_params)
        export_unified_policy(
            weights,
            biases,
            np.array(normalizer.mean),
            np.array(normalizer.std),
            str(policy_cfg["activation"]),
            ckpt_dir / "policy_unified_final.npz",
        )
