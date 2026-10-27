from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.utils import configclass


def _quat_yaw_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (torch.square(quat[:, 2]) + torch.square(quat[:, 3])),
    )


class OmniVelocityCommand(UniformVelocityCommand):
    """Direction-balanced SE(2) velocity commands with deployment-matched slew limits."""

    cfg: OmniVelocityCommandCfg

    def __init__(self, cfg: OmniVelocityCommandCfg, env):
        super().__init__(cfg, env)
        self.target_command_b = torch.zeros_like(self.vel_command_b)
        self.manual_target_b: torch.Tensor | None = None
        self.command_mode = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.axis_tracking_error_sum = torch.zeros(self.num_envs, 3, device=self.device)
        self.axis_tracking_sample_count = torch.zeros(self.num_envs, 3, device=self.device)
        self.standing_motion_error_sum = torch.zeros(self.num_envs, device=self.device)
        self.standing_sample_count = torch.zeros(self.num_envs, device=self.device)
        self.standing_anchor_xy = self.robot.data.root_pos_w[:, :2].clone()
        self.standing_anchor_yaw = _quat_yaw_wxyz(self.robot.data.root_quat_w).clone()
        self.standing_anchor_active = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Raw sums/counts feed the curriculum without averaging moving samples in.
        self.standing_component_sum = torch.zeros(self.num_envs, 4, device=self.device)
        self.standing_stable_samples = torch.zeros(self.num_envs, device=self.device)
        self.standing_stable_duration = torch.zeros(self.num_envs, device=self.device)
        self.standing_max_stable_duration = torch.zeros(self.num_envs, device=self.device)
        self.stopping_attempts = torch.zeros(self.num_envs, device=self.device)
        self.stopping_successes = torch.zeros(self.num_envs, device=self.device)
        self.stopping_time_sum = torch.zeros(self.num_envs, device=self.device)
        self._standing_target_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._stop_pending = torch.zeros_like(self._standing_target_active)
        self._stop_elapsed = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_x"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_y"] = torch.zeros(self.num_envs, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        resolved_ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        count = self.standing_sample_count[resolved_ids].sum().clamp_min(1.0)
        components = self.standing_component_sum[resolved_ids].sum(dim=0) / count
        standing_metrics = dict(zip(
            ("standing_xy_speed_m_s", "standing_yaw_speed_rad_s", "standing_position_drift_m", "standing_heading_drift_rad"),
            components.tolist(),
        ))
        attempts = self.stopping_attempts[resolved_ids].sum()
        successes = self.stopping_successes[resolved_ids].sum()
        standing_metrics.update(
            standing_samples=float(self.standing_sample_count[resolved_ids].sum()),
            standing_stable_fraction=float(self.standing_stable_samples[resolved_ids].sum() / count),
            standing_max_stable_time_s=float(self.standing_max_stable_duration[resolved_ids].mean()),
            stopping_attempts=float(attempts),
            stopping_success_fraction=float(successes / attempts.clamp_min(1.0)),
            stopping_settle_time_s=float(self.stopping_time_sum[resolved_ids].sum() / successes.clamp_min(1.0)),
        )
        self.vel_command_b[resolved_ids] = 0.0
        self.target_command_b[resolved_ids] = 0.0
        metrics = super().reset(resolved_ids)
        metrics.update(standing_metrics)
        self.axis_tracking_error_sum[resolved_ids] = 0.0
        self.axis_tracking_sample_count[resolved_ids] = 0.0
        self.standing_motion_error_sum[resolved_ids] = 0.0
        self.standing_sample_count[resolved_ids] = 0.0
        self.standing_anchor_xy[resolved_ids] = self.robot.data.root_pos_w[resolved_ids, :2]
        self.standing_anchor_yaw[resolved_ids] = _quat_yaw_wxyz(
            self.robot.data.root_quat_w[resolved_ids]
        )
        self.standing_anchor_active[resolved_ids] = False
        for buffer in (self.standing_component_sum, self.standing_stable_samples, self.standing_stable_duration,
                       self.standing_max_stable_duration, self.stopping_attempts, self.stopping_successes,
                       self.stopping_time_sum, self._stop_elapsed):
            buffer[resolved_ids] = 0.0
        self._stop_pending[resolved_ids] = False
        # Initial standing is not a walking-to-stop trial.
        self._standing_target_active[resolved_ids] = self.is_standing_env[resolved_ids]
        return metrics

    def enable_manual_control(self):
        """Start playback at rest and keep keyboard targets across command resampling."""
        self.manual_target_b = torch.zeros_like(self.target_command_b)
        self.target_command_b.zero_()
        self.vel_command_b.zero_()
        self.is_standing_env.fill_(True)

    def set_manual_target(self, velocity: tuple[float, float, float]):
        """Set the requested base-frame x, y and yaw velocities for every environment."""
        if self.manual_target_b is None:
            raise RuntimeError("Manual control has not been enabled")
        self.manual_target_b[:] = self.manual_target_b.new_tensor(velocity)
        self.target_command_b[:] = self.manual_target_b
        self.is_standing_env[:] = torch.linalg.norm(self.manual_target_b, dim=1) < self.cfg.standing_command_threshold

    def _sample_nonzero(self, count: int, value_range, minimum: float) -> torch.Tensor:
        low, high = (float(value) for value in value_range)
        values = torch.empty(count, device=self.device).uniform_(low, high)
        if minimum <= 0.0:
            return values

        small = torch.abs(values) < minimum
        if not torch.any(small):
            return values

        positive_available = high >= minimum
        negative_available = low <= -minimum
        if positive_available and negative_available:
            use_positive = torch.rand(count, device=self.device) < 0.5
            replacement = torch.where(
                use_positive,
                torch.empty(count, device=self.device).uniform_(minimum, high),
                -torch.empty(count, device=self.device).uniform_(minimum, -low),
            )
        elif positive_available:
            replacement = torch.empty(count, device=self.device).uniform_(minimum, high)
        elif negative_available:
            replacement = -torch.empty(count, device=self.device).uniform_(minimum, -low)
        else:
            replacement = torch.zeros(count, device=self.device)
        return torch.where(small, replacement, values)

    def _resample_command(self, env_ids: Sequence[int]):
        count = len(env_ids)
        if count == 0:
            return

        if self.manual_target_b is not None:
            self.target_command_b[env_ids] = self.manual_target_b[env_ids]
            self.is_standing_env[env_ids] = (
                torch.linalg.norm(self.manual_target_b[env_ids], dim=1) < self.cfg.standing_command_threshold
            )
            return

        self.target_command_b[env_ids] = 0.0
        random_values = torch.rand(count, device=self.device)
        standing = random_values < self.cfg.rel_standing_envs
        self.is_standing_env[env_ids] = standing

        moving_random = torch.rand(count, device=self.device)
        cumulative = torch.tensor(
            self.cfg.mode_probabilities,
            dtype=torch.float,
            device=self.device,
        ).cumsum(dim=0)
        modes = torch.bucketize(moving_random.contiguous(), cumulative.contiguous()) + 1
        modes = torch.clamp(modes, max=4)
        modes[standing] = 0
        self.command_mode[env_ids] = modes

        vx = self._sample_nonzero(count, self.cfg.ranges.lin_vel_x, self.cfg.min_command[0])
        vy = self._sample_nonzero(count, self.cfg.ranges.lin_vel_y, self.cfg.min_command[1])
        yaw = self._sample_nonzero(count, self.cfg.ranges.ang_vel_z, self.cfg.min_command[2])

        target = self.target_command_b[env_ids]
        longitudinal = modes == 1
        lateral = modes == 2
        turning = modes == 3
        combined = modes == 4
        target[longitudinal | combined, 0] = vx[longitudinal | combined]
        target[lateral | combined, 1] = vy[lateral | combined]
        target[turning | combined, 2] = yaw[turning | combined]
        self.target_command_b[env_ids] = target

    def _update_command(self):
        target = self.target_command_b
        current = self.vel_command_b
        acceleration = torch.tensor(self.cfg.acceleration, device=self.device).unsqueeze(0)
        deceleration = torch.tensor(self.cfg.deceleration, device=self.device).unsqueeze(0)
        accelerating = (current * target >= 0.0) & (torch.abs(target) > torch.abs(current))
        rate = torch.where(accelerating, acceleration, deceleration)
        max_delta = rate * float(self._env.step_dt)
        self.vel_command_b += torch.clamp(target - current, min=-max_delta, max=max_delta)

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        error_x = torch.abs(self.vel_command_b[:, 0] - self.robot.data.root_lin_vel_b[:, 0])
        error_y = torch.abs(self.vel_command_b[:, 1] - self.robot.data.root_lin_vel_b[:, 1])
        error_yaw = torch.abs(self.vel_command_b[:, 2] - self.robot.data.root_ang_vel_b[:, 2])
        self.metrics["error_vel_x"] += error_x / max_command_step
        self.metrics["error_vel_y"] += error_y / max_command_step
        self.metrics["error_vel_xy"] += torch.sqrt(error_x.square() + error_y.square()) / max_command_step
        self.metrics["error_vel_yaw"] += error_yaw / max_command_step

        errors = torch.stack((error_x, error_y, error_yaw), dim=1)
        thresholds = torch.tensor(self.cfg.tracking_command_thresholds, device=self.device).unsqueeze(0)
        active_axes = torch.abs(self.vel_command_b) >= thresholds
        self.axis_tracking_error_sum += errors * active_axes.float()
        self.axis_tracking_sample_count += active_axes.float()

        self._update_standing_metrics()

    def _update_standing_metrics(self):
        """Track one fixed anchor per stand, plus actual walking-to-stop trials."""

        # Exclude the acceleration ramp of a newly sampled moving command from
        # the standing score. Otherwise low-speed locomotion is incorrectly
        # judged as failed standing before the command has ramped up.
        standing = self.is_standing_env & (
            torch.linalg.norm(self.vel_command_b, dim=1) < self.cfg.standing_command_threshold
        )
        root_xy = self.robot.data.root_pos_w[:, :2]
        root_yaw = _quat_yaw_wxyz(self.robot.data.root_quat_w)
        entering_stand = standing & ~self.standing_anchor_active
        moving = ~standing
        self.standing_anchor_xy[moving | entering_stand] = root_xy[moving | entering_stand]
        self.standing_anchor_yaw[moving | entering_stand] = root_yaw[moving | entering_stand]

        xy_speed = torch.linalg.norm(self.robot.data.root_lin_vel_b[:, :2], dim=1)
        yaw_speed = torch.abs(self.robot.data.root_ang_vel_b[:, 2])
        drift = torch.linalg.norm(root_xy - self.standing_anchor_xy, dim=1)
        yaw_delta = root_yaw - self.standing_anchor_yaw
        heading_error = torch.abs(torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)))
        root_motion = xy_speed + self.cfg.standing_yaw_error_scale * yaw_speed
        root_motion += self.cfg.standing_position_error_scale * drift
        root_motion += self.cfg.standing_heading_error_scale * heading_error
        self.standing_motion_error_sum += root_motion * standing.float()
        self.standing_sample_count += standing.float()
        self.standing_anchor_active.copy_(standing)
        self.standing_component_sum += torch.stack((xy_speed, yaw_speed, drift, heading_error), dim=1) * standing[:, None]

        stable = standing & (xy_speed < self.cfg.standing_stable_xy_speed)
        stable &= (yaw_speed < self.cfg.standing_stable_yaw_speed)
        stable &= (drift < self.cfg.standing_stable_position) & (heading_error < self.cfg.standing_stable_heading)
        stable &= self.robot.data.projected_gravity_b[:, :2].norm(dim=1) < self.cfg.standing_stable_tilt
        self.standing_stable_samples += stable.float()
        self.standing_stable_duration = torch.where(stable, self.standing_stable_duration + self._env.step_dt, 0.0)
        self.standing_max_stable_duration = torch.maximum(self.standing_max_stable_duration, self.standing_stable_duration)

        entering_target = self.is_standing_env & ~self._standing_target_active
        self.stopping_attempts += entering_target.float()
        self._stop_pending |= entering_target
        self._stop_elapsed[entering_target] = 0.0
        self._stop_elapsed += self._stop_pending * self._env.step_dt
        success = self._stop_pending & self.is_standing_env
        success &= self.standing_stable_duration >= self.cfg.stopping_stable_time
        success &= self._stop_elapsed <= self.cfg.stopping_timeout
        self.stopping_successes += success.float()
        self.stopping_time_sum += success * self._stop_elapsed
        self._stop_pending &= ~success & self.is_standing_env & (self._stop_elapsed < self.cfg.stopping_timeout)
        self._standing_target_active.copy_(self.is_standing_env)


@configclass
class OmniVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = OmniVelocityCommand
    limit_ranges: UniformVelocityCommandCfg.Ranges = MISSING
    # Probabilities for longitudinal, lateral, yaw-only and combined modes.
    mode_probabilities: tuple[float, float, float, float] = (0.36, 0.29, 0.21, 0.14)
    min_command: tuple[float, float, float] = (0.08, 0.06, 0.10)
    acceleration: tuple[float, float, float] = (0.80, 0.60, 1.20)
    deceleration: tuple[float, float, float] = (1.20, 0.90, 1.80)
    tracking_command_thresholds: tuple[float, float, float] = (0.05, 0.04, 0.07)
    standing_command_threshold: float = 0.04
    standing_yaw_error_scale: float = 0.25
    standing_position_error_scale: float = 1.0
    standing_heading_error_scale: float = 0.15
    standing_stable_xy_speed: float = 0.04
    standing_stable_yaw_speed: float = 0.10
    standing_stable_position: float = 0.04
    standing_stable_heading: float = 0.10
    standing_stable_tilt: float = 0.10
    stopping_stable_time: float = 0.5
    stopping_timeout: float = 2.0

    def __post_init__(self):
        if min(self.standing_stable_xy_speed, self.standing_stable_yaw_speed, self.standing_stable_position,
               self.standing_stable_heading, self.standing_stable_tilt, self.stopping_stable_time) <= 0:
            raise ValueError("Standing tolerances and stable time must be positive")
        if self.stopping_timeout <= self.stopping_stable_time:
            raise ValueError("Stopping timeout must exceed stable time")
        if len(self.mode_probabilities) != 4 or any(value < 0.0 for value in self.mode_probabilities):
            raise ValueError("mode_probabilities must contain four non-negative values.")
        total_probability = sum(self.mode_probabilities)
        if abs(total_probability - 1.0) > 1.0e-6:
            raise ValueError("mode_probabilities must sum to one.")
        if not 0.0 <= self.rel_standing_envs < 1.0:
            raise ValueError("rel_standing_envs must be in [0, 1).")
        if any(value < 0.0 for value in self.min_command):
            raise ValueError("min_command values must be non-negative.")
        if any(value <= 0.0 for value in self.acceleration + self.deceleration):
            raise ValueError("command acceleration and deceleration must be positive.")
        if len(self.tracking_command_thresholds) != 3 or any(
            value <= 0.0 for value in self.tracking_command_thresholds
        ):
            raise ValueError("tracking_command_thresholds must contain three positive values.")
        if self.standing_command_threshold <= 0.0 or min(
            self.standing_yaw_error_scale,
            self.standing_position_error_scale,
            self.standing_heading_error_scale,
        ) < 0.0:
            raise ValueError("standing thresholds must be positive and yaw scale non-negative.")


# Backward-compatible name for older task configurations.
UniformLevelVelocityCommandCfg = OmniVelocityCommandCfg
