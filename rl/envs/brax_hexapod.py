from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from brax.envs.base import PipelineEnv, State
from brax.io import mjcf
import jax
from jax import numpy as jp

from rl.envs.rewards import locomotion_reward
from rl.envs.terrain import HeightfieldSpec, make_heightfield_xml


@dataclass
class BraxHexapodConfig:
    model_path: Path
    episode_length: int
    action_repeat: int
    action_scale: float
    target_velocity: float
    ctrl_cost: float
    orient_cost: float
    joint_limit_cost: float
    alive_bonus: float
    reset_noise: float
    terrain: str
    heightfield: HeightfieldSpec | None


class BraxHexapodEnv(PipelineEnv):
    def __init__(self, cfg: BraxHexapodConfig, backend: str = "mjx"):
        xml_text = cfg.model_path.read_text(encoding="utf-8")
        if cfg.terrain == "heightfield" and cfg.heightfield is not None:
            xml_text = make_heightfield_xml(xml_text, cfg.heightfield)
        sys = mjcf.loads(xml_text)
        super().__init__(sys=sys, backend=backend, n_frames=cfg.action_repeat)
        self.cfg = cfg
        self.nq_root = 7
        self.nv_root = 6
        self._root_link = 0
        if hasattr(self.sys, "link_names") and "hexapod" in self.sys.link_names:
            self._root_link = self.sys.link_names.index("hexapod")
        self._joint_ranges = self._build_joint_ranges()

    def _build_joint_ranges(self) -> jp.ndarray:
        ranges = []
        for j in range(self.sys.njnt):
            if self.sys.jnt_type[j] != 3:
                continue
            low, high = self.sys.jnt_range[j]
            ranges.append((low, high))
        if not ranges:
            return jp.zeros((0, 2))
        return jp.array(ranges)

    def reset(self, rng: jax.Array) -> State:
        rng, rng1, rng2 = jax.random.split(rng, 3)
        low, high = -self.cfg.reset_noise, self.cfg.reset_noise
        q = self.sys.init_q + jax.random.uniform(
            rng1, (self.sys.q_size(),), minval=low, maxval=high
        )
        qd = high * jax.random.normal(rng2, (self.sys.qd_size(),))
        pipeline_state = self.pipeline_init(q, qd)
        obs = self._get_obs(pipeline_state)
        reward = jp.zeros(())
        done = jp.zeros(())
        metrics = {
            "reward/forward": jp.zeros(()),
            "reward/ctrl": jp.zeros(()),
            "reward/orient": jp.zeros(()),
            "reward/joint": jp.zeros(()),
            "reward/total": jp.zeros(()),
            "state/forward_vel": jp.zeros(()),
            "state/up_z": jp.zeros(()),
        }
        return State(pipeline_state, obs, reward, done, metrics)

    def step(self, state: State, action: jax.Array) -> State:
        action = jp.clip(action, -1.0, 1.0) * self.cfg.action_scale
        pipeline_state0 = state.pipeline_state
        assert pipeline_state0 is not None
        pipeline_state = self.pipeline_step(pipeline_state0, action)
        obs = self._get_obs(pipeline_state)
        reward, metrics = self._get_reward(pipeline_state, action)
        done = jp.zeros(())
        return state.replace(pipeline_state=pipeline_state, obs=obs, reward=reward, done=done, metrics=metrics)

    def _get_obs(self, pipeline_state) -> jax.Array:
        qpos = pipeline_state.q[self.nq_root :]
        qvel = pipeline_state.qd[self.nv_root :]
        base_vel = pipeline_state.qd[: self.nv_root]
        return jp.concatenate([base_vel, qpos, qvel])

    def _get_up_z(self, pipeline_state) -> jax.Array:
        mat = pipeline_state.x.mat[self._root_link]
        up = mat[:, 2]
        return up[2]

    def _joint_limit_cost(self, pipeline_state) -> jax.Array:
        if self._joint_ranges.shape[0] == 0:
            return jp.zeros(())
        qpos = pipeline_state.q[self.nq_root : self.nq_root + self._joint_ranges.shape[0]]
        low = self._joint_ranges[:, 0]
        high = self._joint_ranges[:, 1]
        below = jp.clip(low - qpos, 0.0)
        above = jp.clip(qpos - high, 0.0)
        return jp.mean(below * below + above * above)

    def _get_reward(self, pipeline_state, action: jax.Array):
        forward_vel = pipeline_state.qd[0]
        up_z = self._get_up_z(pipeline_state)
        joint_cost = self._joint_limit_cost(pipeline_state)
        reward, metrics = locomotion_reward(
            jp,
            forward_vel,
            action,
            up_z,
            joint_cost,
            target_velocity=self.cfg.target_velocity,
            ctrl_cost=self.cfg.ctrl_cost,
            orient_cost=self.cfg.orient_cost,
            joint_cost=self.cfg.joint_limit_cost,
            alive_bonus=self.cfg.alive_bonus,
        )
        return reward, metrics
