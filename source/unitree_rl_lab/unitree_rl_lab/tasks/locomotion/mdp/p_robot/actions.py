from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.utils import DelayBuffer, configclass


class MotorResponseJointPositionAction(JointPositionAction):
    """MuJoCo-aligned joint-offset action with delay and first-order response.

    ``cfg.clip`` is interpreted as an offset limit, matching the deployment
    controller: ``q_target = default_q + clamp(action * scale, clip)``.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.raw_action_clip is not None and cfg.raw_action_clip <= 0.0:
            raise ValueError("raw_action_clip must be > 0 when enabled.")
        if cfg.min_command_delay_s < 0.0 or cfg.max_command_delay_s < cfg.min_command_delay_s:
            raise ValueError("Expected 0 <= min_command_delay_s <= max_command_delay_s.")
        if cfg.min_settling_time_s < 0.0 or cfg.max_settling_time_s < cfg.min_settling_time_s:
            raise ValueError("Expected 0 <= min_settling_time_s <= max_settling_time_s.")
        if not 0.0 < cfg.settling_tolerance < 1.0:
            raise ValueError("Expected settling_tolerance to be in (0, 1).")

        # Communication delay is shared by all motors in one environment and is
        # quantized to the physics rate (5 ms for P3).
        # For a first-order response, the remaining step error is exp(-t / tau).
        # Choose tau so the error reaches settling_tolerance at settling_time.
        self._physics_dt = float(env.physics_dt)
        self._min_command_delay_steps = max(
            0,
            math.ceil(cfg.min_command_delay_s / self._physics_dt - 1.0e-9),
        )
        self._max_command_delay_steps = max(
            0,
            math.ceil(cfg.max_command_delay_s / self._physics_dt - 1.0e-9),
        )
        num_command_delay_levels = self._max_command_delay_steps - self._min_command_delay_steps + 1
        if cfg.command_delay_probabilities is None:
            self._command_delay_probabilities = torch.ones(
                num_command_delay_levels,
                device=self.device,
            )
        else:
            if len(cfg.command_delay_probabilities) != num_command_delay_levels:
                raise ValueError(
                    "command_delay_probabilities must have one entry for every physics-step delay level "
                    f"from {self._min_command_delay_steps} to {self._max_command_delay_steps}."
                )
            self._command_delay_probabilities = torch.tensor(
                cfg.command_delay_probabilities,
                dtype=torch.float,
                device=self.device,
            )
            if torch.any(self._command_delay_probabilities < 0.0):
                raise ValueError("command_delay_probabilities cannot contain negative values.")
            if not torch.any(self._command_delay_probabilities > 0.0):
                raise ValueError("command_delay_probabilities must contain at least one positive value.")
        self._command_delay_probabilities /= self._command_delay_probabilities.sum()
        self._min_settling_time_steps = max(
            0,
            math.ceil(cfg.min_settling_time_s / self._physics_dt - 1.0e-9),
        )
        self._max_settling_time_steps = max(
            0,
            math.ceil(cfg.max_settling_time_s / self._physics_dt - 1.0e-9),
        )
        self._settling_time_to_tau = 1.0 / -math.log(cfg.settling_tolerance)
        default_actions = self._get_default_actions()
        self._target_actions = default_actions.clone()
        self._communicated_actions = default_actions.clone()
        self._command_history = default_actions.unsqueeze(0).repeat(
            self._max_command_delay_steps + 1,
            1,
            1,
        )
        self._command_delay_steps = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        self._motor_response_actions = default_actions.clone()
        self._settling_time_s = torch.empty_like(default_actions)
        self._time_constant_s = torch.empty_like(self._settling_time_s)
        self._response_alpha = torch.empty_like(self._settling_time_s)
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        """Convert the newest policy action to the motor's requested position target."""
        if self.cfg.raw_action_clip is not None:
            actions = torch.clamp(actions, -self.cfg.raw_action_clip, self.cfg.raw_action_clip)
        self._raw_actions[:] = actions
        action_offset = actions * self._scale
        if self.cfg.clip is not None:
            action_offset = torch.clamp(
                action_offset,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )
        self._target_actions = action_offset + self._offset

    def apply_actions(self):
        """Advance communication and motor dynamics once per physics step."""
        self._command_history = torch.roll(self._command_history, shifts=1, dims=0)
        self._command_history[0] = self._target_actions
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._communicated_actions[:] = self._command_history[self._command_delay_steps, env_ids]

        self._motor_response_actions += self._response_alpha * (
            self._communicated_actions - self._motor_response_actions
        )
        self._processed_actions[:] = self._motor_response_actions
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    @property
    def command_delay_seconds(self) -> torch.Tensor:
        """Sampled communication delay for each environment."""
        return self._command_delay_steps * self._physics_dt

    @property
    def settling_time_seconds(self) -> torch.Tensor:
        """Sampled 2%-criterion settling time for each environment and motor."""
        return self._settling_time_s

    @property
    def time_constant_seconds(self) -> torch.Tensor:
        """First-order time constant for each environment and motor."""
        return self._time_constant_s

    @property
    def response_alpha(self) -> torch.Tensor:
        """Per-physics-step fraction of the remaining target error removed."""
        return self._response_alpha

    def _get_default_actions(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            default_actions = self._offset.clone()
        else:
            default_actions = torch.full_like(self._processed_actions, float(self._offset))
        return default_actions

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        elif isinstance(env_ids, slice):
            num_envs = self._raw_actions[env_ids].shape[0]
        else:
            num_envs = len(env_ids)

        default_actions = self._get_default_actions()
        self._raw_actions[env_ids] = 0.0
        self._target_actions[env_ids] = default_actions[env_ids]
        self._communicated_actions[env_ids] = default_actions[env_ids]
        self._command_history[:, env_ids] = default_actions[env_ids].unsqueeze(0)
        self._motor_response_actions[env_ids] = default_actions[env_ids]
        self._processed_actions[env_ids] = default_actions[env_ids]
        sampled_delay_levels = torch.multinomial(
            self._command_delay_probabilities,
            num_samples=num_envs,
            replacement=True,
        )
        self._command_delay_steps[env_ids] = sampled_delay_levels + self._min_command_delay_steps
        settling_time_steps = torch.randint(
            self._min_settling_time_steps,
            self._max_settling_time_steps + 1,
            (num_envs, self._processed_actions.shape[1]),
            device=self.device,
        )
        settling_time_s = settling_time_steps * self._physics_dt
        instantaneous_response = settling_time_steps == 0
        time_constant_s = torch.clamp(settling_time_s * self._settling_time_to_tau, min=torch.finfo(torch.float).eps)
        self._settling_time_s[env_ids] = settling_time_s
        self._time_constant_s[env_ids] = torch.where(instantaneous_response, 0.0, time_constant_s)
        self._response_alpha[env_ids] = torch.where(
            instantaneous_response,
            1.0,
            1.0 - torch.exp(-self._physics_dt / time_constant_s),
        )


@configclass
class MotorResponseJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = MotorResponseJointPositionAction
    # This is deliberately separate from the per-joint target-offset clip.
    # It bounds the policy-space action used by the environment and deployment.
    raw_action_clip: float | None = None
    min_command_delay_s: float = 0.0
    max_command_delay_s: float = 0.010
    command_delay_probabilities: tuple[float, ...] | None = None
    min_settling_time_s: float = 0.0
    max_settling_time_s: float = 0.005
    settling_tolerance: float = 0.02


class DeploymentLatencyJointPositionAction(JointPositionAction):
    """Joint-offset action with independent inference and communication latency."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if cfg.raw_action_clip is not None and cfg.raw_action_clip <= 0.0:
            raise ValueError("raw_action_clip must be > 0 when enabled.")
        if cfg.min_inference_delay < 0 or cfg.max_inference_delay < cfg.min_inference_delay:
            raise ValueError("Expected 0 <= min_inference_delay <= max_inference_delay.")
        if cfg.min_communication_delay < 0 or cfg.max_communication_delay < cfg.min_communication_delay:
            raise ValueError("Expected 0 <= min_communication_delay <= max_communication_delay.")

        default_actions = self._get_default_actions()
        self._pending_actions = default_actions.clone()
        self._inferred_actions = default_actions.clone()
        self._inference_delay = torch.empty(self.num_envs, dtype=torch.long, device=self.device)
        self._inference_elapsed = torch.zeros_like(self._inference_delay)
        self._communication_delay = torch.empty(self.num_envs, dtype=torch.int, device=self.device)
        self._communication_buffer = DelayBuffer(
            cfg.max_communication_delay,
            self.num_envs,
            device=self.device,
        )
        self.reset()

    def process_actions(self, actions: torch.Tensor):
        if self.cfg.raw_action_clip is not None:
            actions = torch.clamp(actions, -self.cfg.raw_action_clip, self.cfg.raw_action_clip)
        self._raw_actions[:] = actions
        action_offset = actions * self._scale
        if self.cfg.clip is not None:
            action_offset = torch.clamp(action_offset, min=self._clip[:, :, 0], max=self._clip[:, :, 1])
        self._pending_actions[:] = action_offset + self._offset
        self._inference_elapsed.zero_()

    def apply_actions(self):
        inference_ready = self._inference_elapsed >= self._inference_delay
        self._inferred_actions[inference_ready] = self._pending_actions[inference_ready]
        self._inference_elapsed += 1
        self._processed_actions[:] = self._communication_buffer.compute(self._inferred_actions)
        self._asset.set_joint_position_target(self._processed_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
            num_envs = self.num_envs
        elif isinstance(env_ids, slice):
            num_envs = self._raw_actions[env_ids].shape[0]
        else:
            num_envs = len(env_ids)

        default_actions = self._get_default_actions()
        self._raw_actions[env_ids] = 0.0
        self._pending_actions[env_ids] = default_actions[env_ids]
        self._inferred_actions[env_ids] = default_actions[env_ids]
        self._processed_actions[env_ids] = default_actions[env_ids]
        self._inference_elapsed[env_ids] = 0
        self._inference_delay[env_ids] = torch.randint(
            self.cfg.min_inference_delay,
            self.cfg.max_inference_delay + 1,
            (num_envs,),
            device=self.device,
        )
        communication_delay = torch.randint(
            self.cfg.min_communication_delay,
            self.cfg.max_communication_delay + 1,
            (num_envs,),
            dtype=torch.int,
            device=self.device,
        )
        self._communication_delay[env_ids] = communication_delay
        self._communication_buffer.set_time_lag(communication_delay, env_ids)
        self._communication_buffer.reset(env_ids)

    def _get_default_actions(self) -> torch.Tensor:
        if isinstance(self._offset, torch.Tensor):
            return self._offset.clone()
        return torch.full_like(self._processed_actions, float(self._offset))

    @property
    def inference_delay_steps(self) -> torch.Tensor:
        return self._inference_delay

    @property
    def communication_delay_steps(self) -> torch.Tensor:
        return self._communication_delay


@configclass
class DeploymentLatencyJointPositionActionCfg(JointPositionActionCfg):
    class_type: type = DeploymentLatencyJointPositionAction
    raw_action_clip: float | None = None
    min_inference_delay: int = 0
    max_inference_delay: int = 0
    min_communication_delay: int = 0
    max_communication_delay: int = 0
