from __future__ import annotations

from typing import Any, Tuple


def locomotion_reward(
    np_mod: Any,
    forward_vel: Any,
    action: Any,
    up_vector_z: Any,
    joint_limit_cost: Any,
    target_velocity: float,
    ctrl_cost: float,
    orient_cost: float,
    joint_cost: float,
    alive_bonus: float,
) -> Tuple[Any, dict]:
    vel_reward = 1.0 - np_mod.square(forward_vel - target_velocity)
    ctrl_penalty = ctrl_cost * np_mod.mean(np_mod.square(action))
    orient_penalty = orient_cost * np_mod.square(1.0 - up_vector_z)
    joint_penalty = joint_cost * joint_limit_cost

    reward = vel_reward - ctrl_penalty - orient_penalty - joint_penalty + alive_bonus
    metrics = {
        "reward/forward": vel_reward,
        "reward/ctrl": -ctrl_penalty,
        "reward/orient": -orient_penalty,
        "reward/joint": -joint_penalty,
        "reward/total": reward,
        "state/forward_vel": forward_vel,
        "state/up_z": up_vector_z,
    }
    return reward, metrics
