"""P-specific correction for the stationary-foot clearance loophole."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply_inverse, yaw_quat


def _standing_mask(env, command_name: str) -> torch.Tensor:
    """Exclude moving-command ramps and intentional slow/turning commands."""
    term = env.command_manager.get_term(command_name)
    return term.is_standing_env & (
        torch.linalg.vector_norm(term.command, dim=1) < term.cfg.standing_command_threshold
    )


def standing_base_xy_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    velocity = env.scene[asset_cfg.name].data.root_lin_vel_b[:, :2]
    return velocity.square().sum(dim=1) * _standing_mask(env, command_name)


def standing_base_yaw_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    velocity = env.scene[asset_cfg.name].data.root_ang_vel_b[:, 2]
    return velocity.square() * _standing_mask(env, command_name)


def standing_joint_vel_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
    return velocity.square().sum(dim=1) * _standing_mask(env, command_name)


def standing_action_rate_l2(env, command_name: str):
    delta = env.action_manager.action - env.action_manager.prev_action
    return delta.square().sum(dim=1) * _standing_mask(env, command_name)


def standing_position_drift_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    term = env.command_manager.get_term(command_name)
    delta = env.scene[asset_cfg.name].data.root_pos_w[:, :2] - term.standing_anchor_xy
    return delta.square().sum(dim=1) * _standing_mask(env, command_name) * term.standing_anchor_active


def standing_heading_drift_l2(env, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    from .velocity_command import _quat_yaw_wxyz

    term = env.command_manager.get_term(command_name)
    delta = _quat_yaw_wxyz(env.scene[asset_cfg.name].data.root_quat_w) - term.standing_anchor_yaw
    error = torch.atan2(torch.sin(delta), torch.cos(delta))
    return error.square() * _standing_mask(env, command_name) * term.standing_anchor_active


def standing_contact_spin_l2(env, command_name: str, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg):
    """Penalize world-vertical foot spin only while that foot has ground contact."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1) > 1.0
    yaw_speed = env.scene[asset_cfg.name].data.body_ang_vel_w[:, asset_cfg.body_ids, 2]
    return (yaw_speed.square() * contact).sum(dim=1) * _standing_mask(env, command_name)


def track_lin_vel_xy_progress(
    env,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_threshold: float = 0.05,
    gait_reward_name: str | None = None,
    gait_exploration_fraction: float = 0.25,
) -> torch.Tensor:
    """Reward velocity tracking in proportion to actual commanded-direction motion.

    Standing earns zero for a moving planar command. A zero planar command
    retains the usual stationary tracking objective, including yaw-only motion.
    When gait_reward_name is configured, moving-command reward interpolates
    from an exploration floor to full reward using recent completed-step quality.
    """
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    velocity = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w)[:, :2]
    command_sq = command.square().sum(dim=1)
    error_sq = (command - velocity).square().sum(dim=1)
    tracking = torch.exp(-error_sq / std**2)
    progress = ((command * velocity).sum(dim=1) / command_sq.clamp_min(1.0e-6)).clamp(0.0, 1.0)
    moving = command_sq > command_threshold**2
    result = torch.where(moving, tracking * progress, tracking)
    if gait_reward_name is not None:
        if not 0 < gait_exploration_fraction <= 1:
            raise ValueError("Gait exploration fraction must be in (0, 1]")
        from .foot_trajectory_rewards import _alternating_state

        quality = _alternating_state(env, gait_reward_name).recent_quality
        factor = gait_exploration_fraction + (1.0 - gait_exploration_fraction) * quality
        result = torch.where(moving, result * factor, result)
    return result


def foot_clearance_during_swing(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
    period: float,
    offset: tuple[float, float],
    threshold: float,
    command_name: str,
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    foot_speed = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    clearance = torch.exp(-torch.square(foot_height - target_height) / std)
    moving_foot = torch.tanh(tanh_mult * foot_speed)

    phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    phase = (phase + torch.as_tensor(offset, device=env.device)) % 1.0
    swing = phase >= threshold
    airborne = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return torch.sum(clearance * moving_foot * swing * airborne, dim=1) * commanded
