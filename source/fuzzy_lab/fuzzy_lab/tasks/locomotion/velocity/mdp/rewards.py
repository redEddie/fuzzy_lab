from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward

def stand_still(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward standing still.

    This function rewards the agent for standing still when the command is zero.
    It penalizes joint position deviations from the default position only when the command is below a threshold.

    Args:
        env: The environment instance.
        command_name: The name of the command to check.
        threshold: The threshold below which the command is considered zero.
        asset_cfg: The configuration for the asset.

    Returns:
        A tensor containing the reward for standing still.
    """
    # Get the command
    command = env.command_manager.get_command(command_name)
    # Get the joint positions
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    # Get the default joint positions
    default_joint_pos = env.scene[asset_cfg.name].data.default_joint_pos[:, asset_cfg.joint_ids]
    # Compute the deviation from default position
    deviation = torch.abs(joint_pos - default_joint_pos)
    # Sum the deviations
    total_deviation = torch.sum(deviation, dim=1)
    # Check if command is below threshold
    is_zero_command = torch.norm(command[:, :2], dim=1) < threshold
    # Apply penalty only when command is zero
    reward = total_deviation * is_zero_command.float()
    # 반환값 검증 및 경고
    if torch.isnan(reward).any():
        print("[WARN] stand_still: NaN detected!", reward)
    if torch.isinf(reward).any():
        print("[WARN] stand_still: Inf detected!", reward)
    if (reward < 0).any():
        print("[WARN] stand_still: Negative value detected!", reward)
    if reward.abs().max() > 1e3:
        print("[WARN] stand_still: Large value detected!", reward.abs().max())
    return reward