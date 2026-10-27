"""Encoder-frame actions and residual parameter randomization for PACE."""

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from .actions import DeploymentLatencyJointPositionAction, DeploymentLatencyJointPositionActionCfg


class PaceJointPositionAction(DeploymentLatencyJointPositionAction):
    """Keep offsets in the encoder frame, while reset states are physical angles."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        if not cfg.use_default_offset:
            raise ValueError("PACE actions require use_default_offset=True")
        bias = torch.zeros_like(self._asset.data.default_joint_pos)
        for actuator in self._asset.actuators.values():
            bias[:, actuator.joint_indices] = actuator.encoder_bias
        self._offset = self._offset - bias[:, self._joint_ids]
        self.reset()


@configclass
class PaceJointPositionActionCfg(DeploymentLatencyJointPositionActionCfg):
    class_type: type = PaceJointPositionAction


def randomize_pace_joint_parameters(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    armature_scale_range: tuple[float, float] = (0.9, 1.1),
    friction_scale_range: tuple[float, float] = (0.9, 1.1),
    viscous_friction_scale_range: tuple[float, float] = (0.9, 1.1),
):
    """Sample around fitted defaults; retain equal static/dynamic coefficients.

    These ranges are engineering choices, not confidence intervals from PACE.
    Friction and armature affect PhysX; encoder bias and torque delay stay fixed.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(asset.num_instances, device=asset.device)
    joint_ids = asset_cfg.joint_ids

    def sample(default, bounds):
        low, high = bounds
        if not 0.0 <= low <= high:
            raise ValueError(f"Invalid PACE scale range: {bounds}")
        nominal = default[env_ids][:, joint_ids]
        return nominal * torch.empty_like(nominal).uniform_(low, high)

    armature = sample(asset.data.default_joint_armature, armature_scale_range)
    friction = sample(asset.data.default_joint_friction_coeff, friction_scale_range)
    viscous = sample(asset.data.default_joint_viscous_friction_coeff, viscous_friction_scale_range)
    asset.write_joint_armature_to_sim(armature, joint_ids=joint_ids, env_ids=env_ids)
    # Lower dynamic first, as in PACE, to avoid static < previous dynamic.
    asset.write_joint_dynamic_friction_coefficient_to_sim(0.0, joint_ids=joint_ids, env_ids=env_ids)
    asset.write_joint_friction_coefficient_to_sim(
        friction, joint_dynamic_friction_coeff=friction, joint_viscous_friction_coeff=viscous,
        joint_ids=joint_ids, env_ids=env_ids,
    )
