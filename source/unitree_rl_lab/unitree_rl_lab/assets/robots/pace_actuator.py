"""PACE DC motor curve, encoder-frame PD and torque delay.

The bias sign and delay location follow the local ETH PACE implementation
(Apache-2.0, Filip Bjelonic, 2025). DCMotor supplies the same four-quadrant
velocity-dependent saturation used by the P1 identification configuration.
Friction and armature are configured in PhysX, never subtracted here again.
"""

from collections.abc import Sequence

import torch

from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class PaceDCMotor(DCMotor):
    def __init__(self, cfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if not isinstance(cfg.delay_steps, int) or cfg.delay_steps < 0:
            raise ValueError("delay_steps must be a nonnegative integer")
        self.encoder_bias = self._parse_joint_parameter(cfg.encoder_bias, 0.0)
        self.torques_delay_buffer = DelayBuffer(cfg.delay_steps, self._num_envs, device=self._device)
        self.torques_delay_buffer.set_time_lag(cfg.delay_steps)

    def reset(self, env_ids: Sequence[int]):
        super().reset(env_ids)
        self.torques_delay_buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        # Local PACE code uses q_encoder = q_sim - bias (its prose says "added").
        result = super().compute(control_action, joint_pos - self.encoder_bias, joint_vel)
        self.applied_effort = self.torques_delay_buffer.compute(result.joint_efforts)
        result.joint_efforts = self.applied_effort
        return result


@configclass
class PaceDCMotorCfg(DCMotorCfg):
    class_type: type = PaceDCMotor
    encoder_bias: dict[str, float] | float = 0.0
    delay_steps: int = 0


# Keep configuration imports from the initial migration usable.
PacePDActuator = PaceDCMotor
PacePDActuatorCfg = PaceDCMotorCfg
