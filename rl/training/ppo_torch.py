from __future__ import annotations

from dataclasses import dataclass
import multiprocessing as mp
import os
from pathlib import Path
from typing import Dict, List, Tuple
import time

import numpy as np

from rl.common.logger import CsvLogger
from rl.common.seeding import seed_all, fold_in_seed
from rl.envs.mujoco_hexapod import HexapodEnvConfig, MujocoHexapodEnv
from rl.envs.terrain import HeightfieldSpec
from rl.policy.unified_policy import export_unified_policy, extract_torch_mlp


def _env_worker(cfg: HexapodEnvConfig, seeds: list, pipe: mp.connection.Connection) -> None:
    """Worker process: owns a slice of environments and steps them on request."""
    rngs = [np.random.default_rng(s) for s in seeds]
    envs = [MujocoHexapodEnv(HexapodEnvConfig(**cfg.__dict__)) for _ in seeds]
    while True:
        cmd, data = pipe.recv()
        if cmd == "reset":
            pipe.send([env.reset(rng) for env, rng in zip(envs, rngs)])
        elif cmd == "step":
            results = []
            for env, rng, action in zip(envs, rngs, data):
                ob, rew, done, info = env.step(action)
                if done:
                    ob = env.reset(rng)
                results.append((ob, rew, done, info))
            pipe.send(results)
        elif cmd == "close":
            return


@dataclass
class TorchPPOConfig:
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


class RunningMeanStd:
    def __init__(self, shape: int):
        self.mean = np.zeros(shape, dtype=np.float32)
        self.var = np.ones(shape, dtype=np.float32)
        self.count = 1e-4

    def update(self, x: np.ndarray) -> None:
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, mean: np.ndarray, var: np.ndarray, count: int) -> None:
        delta = mean - self.mean
        total_count = self.count + count
        new_mean = self.mean + delta * count / total_count
        m_a = self.var * self.count
        m_b = var * count
        m2 = m_a + m_b + delta * delta * self.count * count / total_count
        new_var = m2 / total_count
        self.mean = new_mean
        self.var = new_var
        self.count = total_count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


class MujocoVecEnv:
    def __init__(self, cfg: HexapodEnvConfig, num_envs: int, seed: int):
        self.num_envs = num_envs
        num_workers = min(num_envs, os.cpu_count() or 1)

        # Sizes from a throwaway env (cheap: no rollout yet)
        _tmp = MujocoHexapodEnv(cfg)
        self.observation_size = _tmp.observation_size
        self.action_size = _tmp.action_size
        del _tmp

        seeds = [fold_in_seed(seed, i) for i in range(num_envs)]
        chunks = [list(range(i, num_envs, num_workers)) for i in range(num_workers)]

        self._pipes: List[mp.connection.Connection] = []
        self._chunk_sizes: List[int] = []
        self._procs: List[mp.Process] = []
        for chunk in chunks:
            if not chunk:
                continue
            chunk_seeds = [seeds[i] for i in chunk]
            parent_pipe, child_pipe = mp.Pipe()
            proc = mp.Process(
                target=_env_worker,
                args=(cfg, chunk_seeds, child_pipe),
                daemon=True,
            )
            proc.start()
            child_pipe.close()
            self._pipes.append(parent_pipe)
            self._chunk_sizes.append(len(chunk))
            self._procs.append(proc)

    def reset(self) -> np.ndarray:
        for pipe in self._pipes:
            pipe.send(("reset", None))
        obs = []
        for pipe in self._pipes:
            obs.extend(pipe.recv())
        return np.stack(obs, axis=0)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, float]]]:
        offset = 0
        for pipe, size in zip(self._pipes, self._chunk_sizes):
            pipe.send(("step", actions[offset:offset + size]))
            offset += size
        obs, rewards, dones, infos = [], [], [], []
        for pipe in self._pipes:
            for ob, rew, done, info in pipe.recv():
                obs.append(ob)
                rewards.append(rew)
                dones.append(done)
                infos.append(info)
        return (
            np.stack(obs, axis=0),
            np.asarray(rewards, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            infos,
        )

    def close(self) -> None:
        for pipe in self._pipes:
            try:
                pipe.send(("close", None))
            except Exception:
                pass
        for proc in self._procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


class PolicyValueNet:
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: List[int], activation: str, log_std_init: float):
        import torch
        import torch.nn as nn

        class _Module(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList()
                last = obs_dim
                for hidden in hidden_sizes:
                    self.layers.append(nn.Linear(last, hidden))
                    last = hidden
                self.policy_mean = nn.Linear(last, action_dim)
                self.value_head = nn.Linear(last, 1)
                self.log_std = nn.Parameter(torch.full((action_dim,), log_std_init))

        self._module = _Module()
        self.torch = torch
        self.activation = activation

    def forward(self, obs):
        x = obs
        for layer in self._module.layers:
            x = _activate(self.torch, layer(x), self.activation)
        mean = self._module.policy_mean(x)
        value = self._module.value_head(x).squeeze(-1)
        return mean, value

    def parameters(self):
        return self._module.parameters()

    def state_dict(self):
        return self._module.state_dict()

    @property
    def log_std(self):
        return self._module.log_std

    def to(self, device):
        self._module.to(device)
        return self


def train_torch(
    env_cfg: HexapodEnvConfig,
    ppo_cfg: TorchPPOConfig,
    policy_cfg: Dict[str, float | List[int] | str],
    log_dir: Path,
    seed: int,
    export_unified: bool,
    eval_interval: int,
    checkpoint_interval: int,
) -> None:
    import torch

    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vec_env = MujocoVecEnv(env_cfg, ppo_cfg.num_envs, seed)
    obs_dim = vec_env.observation_size
    act_dim = vec_env.action_size
    model = PolicyValueNet(
        obs_dim,
        act_dim,
        hidden_sizes=list(policy_cfg["hidden_sizes"]),
        activation=str(policy_cfg["activation"]),
        log_std_init=float(policy_cfg["log_std_init"]),
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=ppo_cfg.learning_rate)
    rms = RunningMeanStd(obs_dim)

    logger = CsvLogger(log_dir)
    logger.log_config({
        "backend": "mujoco",
        "seed": seed,
        "policy": policy_cfg,
        "ppo": ppo_cfg.__dict__,
        "env": env_cfg.__dict__,
    })

    obs = vec_env.reset()
    global_step = 0
    last_eval = 0
    last_checkpoint = 0

    while global_step < ppo_cfg.total_steps:
        rollout = _collect_rollout(model, rms, vec_env, obs, ppo_cfg, device)
        obs = rollout["last_obs"]
        global_step += rollout["steps"]

        if ppo_cfg.normalize_obs:
            rms.update(rollout["obs"])

        _ppo_update(model, optimizer, rms, rollout, ppo_cfg, device)
        logger.log(global_step, {"train/mean_reward": rollout["mean_reward"]})

        if global_step - last_eval >= eval_interval:
            eval_return = _evaluate(model, rms, vec_env, episodes=8, device=device)
            logger.log(global_step, {"eval/episode_return": eval_return})
            last_eval = global_step

        if global_step - last_checkpoint >= checkpoint_interval:
            _save_checkpoint(model, rms, log_dir, global_step, policy_cfg, export_unified)
            last_checkpoint = global_step

    _save_checkpoint(model, rms, log_dir, global_step, policy_cfg, export_unified)
    vec_env.close()


def _collect_rollout(model, rms: RunningMeanStd, env: MujocoVecEnv, obs: np.ndarray, cfg: TorchPPOConfig, device):
    import torch

    obs_buf = []
    actions_buf = []
    logp_buf = []
    values_buf = []
    rewards_buf = []
    dones_buf = []

    for _ in range(cfg.rollout_length):
        obs_t = torch.tensor(_normalize(obs, rms, cfg.normalize_obs), dtype=torch.float32, device=device)
        mean, value = model.forward(obs_t)
        std = torch.exp(model.log_std)
        dist = torch.distributions.Normal(mean, std)
        action_pre = dist.rsample()
        action = torch.tanh(action_pre)
        log_prob = dist.log_prob(action_pre) - torch.log(1.0 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        next_obs, reward, done, _ = env.step(action.detach().cpu().numpy())

        obs_buf.append(obs)
        actions_buf.append(action.detach().cpu().numpy())
        logp_buf.append(log_prob.detach().cpu().numpy())
        values_buf.append(value.detach().cpu().numpy())
        rewards_buf.append(reward)
        dones_buf.append(done)
        obs = next_obs

    obs_buf = np.asarray(obs_buf)
    actions_buf = np.asarray(actions_buf)
    logp_buf = np.asarray(logp_buf)
    values_buf = np.asarray(values_buf)
    rewards_buf = np.asarray(rewards_buf)
    dones_buf = np.asarray(dones_buf)

    last_obs_t = torch.tensor(_normalize(obs, rms, cfg.normalize_obs), dtype=torch.float32, device=device)
    with torch.no_grad():
        _, last_value = model.forward(last_obs_t)
    last_value = last_value.detach().cpu().numpy()

    adv, returns = _compute_gae(rewards_buf, values_buf, dones_buf, last_value, cfg.gamma, cfg.gae_lambda)

    batch = {
        "obs": obs_buf.reshape(-1, obs.shape[-1]),
        "actions": actions_buf.reshape(-1, actions_buf.shape[-1]),
        "logp": logp_buf.reshape(-1),
        "values": values_buf.reshape(-1),
        "returns": returns.reshape(-1),
        "advantages": adv.reshape(-1),
        "last_obs": obs,
        "steps": cfg.rollout_length * env.num_envs,
        "mean_reward": float(np.mean(rewards_buf)),
    }
    return batch


def _ppo_update(model, optimizer, rms: RunningMeanStd, batch, cfg: TorchPPOConfig, device) -> None:
    import torch

    obs = torch.tensor(_normalize(batch["obs"], rms, cfg.normalize_obs), dtype=torch.float32, device=device)
    actions = torch.tensor(batch["actions"], dtype=torch.float32, device=device)
    old_logp = torch.tensor(batch["logp"], dtype=torch.float32, device=device)
    returns = torch.tensor(batch["returns"], dtype=torch.float32, device=device)
    advantages = torch.tensor(batch["advantages"], dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    batch_size = obs.shape[0]
    for _ in range(cfg.update_epochs):
        indices = np.random.permutation(batch_size)
        for start in range(0, batch_size, cfg.minibatch_size):
            end = start + cfg.minibatch_size
            mb_idx = indices[start:end]

            mb_obs = obs[mb_idx]
            mb_actions = actions[mb_idx]
            mb_old_logp = old_logp[mb_idx]
            mb_returns = returns[mb_idx]
            mb_adv = advantages[mb_idx]

            mean, value = model.forward(mb_obs)
            std = torch.exp(model.log_std)
            dist = torch.distributions.Normal(mean, std)
            action_pre = torch.atanh(torch.clamp(mb_actions, -0.999, 0.999))
            log_prob = dist.log_prob(action_pre) - torch.log(1.0 - mb_actions.pow(2) + 1e-6)
            log_prob = log_prob.sum(-1)

            ratio = torch.exp(log_prob - mb_old_logp)
            pg_loss1 = ratio * mb_adv
            pg_loss2 = torch.clamp(ratio, 1.0 - cfg.clip_epsilon, 1.0 + cfg.clip_epsilon) * mb_adv
            policy_loss = -torch.min(pg_loss1, pg_loss2).mean()

            value_loss = 0.5 * (mb_returns - value).pow(2).mean()
            entropy = dist.entropy().sum(-1).mean()

            loss = policy_loss + cfg.vf_coef * value_loss - cfg.ent_coef * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()


def _compute_gae(rewards, values, dones, last_value, gamma, lam):
    T, N = rewards.shape
    adv = np.zeros((T, N), dtype=np.float32)
    last_adv = np.zeros((N,), dtype=np.float32)
    for t in reversed(range(T)):
        mask = 1.0 - dones[t]
        delta = rewards[t] + gamma * last_value * mask - values[t]
        last_adv = delta + gamma * lam * mask * last_adv
        adv[t] = last_adv
        last_value = values[t]
    returns = adv + values
    return adv, returns


def _normalize(obs: np.ndarray, rms: RunningMeanStd, normalize: bool) -> np.ndarray:
    if not normalize:
        return obs
    return (obs - rms.mean) / (rms.std + 1e-8)


def _evaluate(model, rms: RunningMeanStd, env: MujocoVecEnv, episodes: int, device) -> float:
    import torch

    returns = []
    for _ in range(episodes):
        obs = env.reset()
        finished = np.zeros(env.num_envs, dtype=bool)
        ret = np.zeros(env.num_envs, dtype=np.float32)
        while not finished.all():
            obs_t = torch.tensor(_normalize(obs, rms, True), dtype=torch.float32, device=device)
            with torch.no_grad():
                mean, _ = model.forward(obs_t)
            action = torch.tanh(mean).cpu().numpy()
            obs, reward, done, _ = env.step(action)
            ret += reward * ~finished
            finished |= done.astype(bool)
        returns.append(float(np.mean(ret)))
    return float(np.mean(returns))


def _save_checkpoint(model, rms: RunningMeanStd, log_dir: Path, step: int, policy_cfg: Dict[str, float | List[int] | str], export_unified: bool) -> None:
    import torch

    ckpt_dir = log_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"ppo_torch_{step}.pt"
    torch.save({
        "model": model.state_dict(),
        "log_std": model.log_std.detach().cpu().numpy(),
        "obs_mean": rms.mean,
        "obs_std": rms.std,
    }, ckpt_path)

    if export_unified:
        weights, biases = extract_torch_mlp(model)
        export_unified_policy(
            weights,
            biases,
            rms.mean,
            rms.std,
            str(policy_cfg["activation"]),
            ckpt_dir / f"policy_unified_{step}.npz",
        )


def _activate(torch, x, activation: str):
    if activation == "tanh":
        return torch.tanh(x)
    if activation == "relu":
        return torch.relu(x)
    raise ValueError(f"Unsupported activation: {activation}")
