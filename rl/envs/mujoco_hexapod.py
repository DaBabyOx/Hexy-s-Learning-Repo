from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from rl.envs.rewards import locomotion_reward
from rl.envs.terrain import HeightfieldSpec, make_heightfield_xml


@dataclass
class HexapodEnvConfig:
    model_path: Path
    episode_length: int
    action_repeat: int
    target_velocity: float
    ctrl_cost: float
    orient_cost: float
    joint_limit_cost: float
    alive_bonus: float
    reset_noise: float
    terrain: str
    heightfield: HeightfieldSpec | None


class MujocoHexapodEnv:
    def __init__(self, cfg: HexapodEnvConfig):
        try:
            import mujoco
        except ImportError as exc:
            raise ImportError("MuJoCo is required for the PyTorch backend.") from exc

        self._mujoco = mujoco
        self.cfg = cfg
        xml_text = cfg.model_path.read_text(encoding="utf-8")
        if cfg.terrain == "heightfield" and cfg.heightfield is not None:
            xml_text = make_heightfield_xml(xml_text, cfg.heightfield)
            self.model = mujoco.MjModel.from_xml_string(xml_text)
        else:
            self.model = mujoco.MjModel.from_xml_path(str(cfg.model_path))
        self.data = mujoco.MjData(self.model)
        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hexapod")
        if self.body_id < 0:
            self.body_id = 1
        self.step_count = 0
        self.nq_root = 7
        self.nv_root = 6
        self.joint_ranges, self.joint_qpos_ids = self._build_joint_ranges()

    def _build_joint_ranges(self) -> tuple[np.ndarray, np.ndarray]:
        ranges = []
        qpos_ids = []
        for j in range(self.model.njnt):
            if self.model.jnt_type[j] != self._mujoco.mjtJoint.mjJNT_HINGE:
                continue
            low, high = self.model.jnt_range[j]
            ranges.append((low, high))
            qpos_ids.append(self.model.jnt_qposadr[j])
        return (
            np.asarray(ranges, dtype=np.float32),
            np.asarray(qpos_ids, dtype=np.int32),
        )

    @property
    def action_size(self) -> int:
        return self.model.nu

    @property
    def observation_size(self) -> int:
        return self._get_obs().shape[0]

    def reset(self, rng: np.random.Generator) -> np.ndarray:
        self.step_count = 0
        if self.model.nkey > 0:
            self.data.qpos[:] = self.model.key_qpos[0]
        else:
            self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        if self.cfg.reset_noise > 0.0:
            q_noise = rng.uniform(
                low=-self.cfg.reset_noise, high=self.cfg.reset_noise, size=self.model.nq
            )
            qd_noise = rng.normal(scale=self.cfg.reset_noise, size=self.model.nv)
            self.data.qpos[:] = self.data.qpos + q_noise
            self.data.qvel[:] = self.data.qvel + qd_noise
        self._mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        action = np.clip(action, -1.0, 1.0)
        for _ in range(self.cfg.action_repeat):
            self.data.ctrl[:] = action
            self._mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        reward, metrics = self._get_reward(action)
        done = self.step_count >= self.cfg.episode_length
        return obs, float(reward), done, metrics

    def _get_obs(self) -> np.ndarray:
        qpos = self.data.qpos[self.nq_root :]
        qvel = self.data.qvel[self.nv_root :]
        base_vel = self.data.qvel[: self.nv_root]
        return np.concatenate([base_vel, qpos, qvel]).astype(np.float32)

    def _get_up_z(self) -> float:
        mat = self.data.xmat[self.body_id].reshape(3, 3)
        up = mat[:, 2]
        return float(up[2])

    def _joint_limit_cost(self) -> float:
        if self.joint_ranges.size == 0:
            return 0.0
        qpos = self.data.qpos[self.joint_qpos_ids]
        low = self.joint_ranges[:, 0]
        high = self.joint_ranges[:, 1]
        below = np.clip(low - qpos, 0.0, None)
        above = np.clip(qpos - high, 0.0, None)
        return float(np.mean(below * below + above * above))

    def _get_reward(self, action: np.ndarray) -> Tuple[float, Dict[str, float]]:
        forward_vel = float(self.data.qvel[0])
        up_z = self._get_up_z()
        joint_cost = self._joint_limit_cost()
        reward, metrics = locomotion_reward(
            np,
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
        metrics["state/step"] = float(self.step_count)
        return float(reward), {k: float(v) for k, v in metrics.items()}
