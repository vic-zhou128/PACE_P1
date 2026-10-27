"""P asset with all 12 fitted joint parameter sets, mapped by exact name."""

import re

from isaaclab.utils.string import resolve_matching_names_values

from .p import P_CFG
from .p_pace_parameters import JOINT_NAME_MAP, P1_DELAY_STEPS, P1_MOTOR_LIMITS, P1_PARAMETERS
from .pace_actuator import PaceDCMotorCfg


def _group_parameters(parameter, expressions):
    return {
        name: value for name, value in P1_PARAMETERS[parameter].items()
        if any(re.fullmatch(pattern, name) for pattern in expressions)
    }


P_PACE_CFG = P_CFG.copy()
P_PACE_CFG.spawn.asset_path = "/home/pc/unitree_usd/urdf_803_nobattery/urdf/p1.urdf"
P_PACE_CFG.joint_sdk_names = list(JOINT_NAME_MAP.values())
# The old home angles are encoder-frame commands. PACE initializes physical
# positions with initial_encoder_position + bias (cma_es.py:update_simulator).
_, names, values = resolve_matching_names_values(P_CFG.init_state.joint_pos, P_PACE_CFG.joint_sdk_names)
P_PACE_CFG.init_state.joint_pos = {
    name: value + P1_PARAMETERS["encoder_bias"][name] for name, value in zip(names, values)
}
P_PACE_CFG.actuators = {}
for group, old in P_CFG.actuators.items():
    limits = P1_MOTOR_LIMITS["knee_pitch" if group == "knee" else group]
    P_PACE_CFG.actuators[group] = PaceDCMotorCfg(
        joint_names_expr=old.joint_names_expr.copy(),
        saturation_effort=limits["saturation_effort"],
        effort_limit=limits["effort_limit"],
        effort_limit_sim=limits["effort_limit"],
        velocity_limit=limits["velocity_limit"],
        # Like the identification config, leave the solver speed cap at the
        # URDF value; the motor's no-load speed is enforced through its curve.
        velocity_limit_sim=None,
        stiffness=180.0,
        damping=10.0,
        armature=_group_parameters("armature", old.joint_names_expr),
        friction=_group_parameters("friction", old.joint_names_expr),
        dynamic_friction=_group_parameters("friction", old.joint_names_expr),
        viscous_friction=_group_parameters("viscous_friction", old.joint_names_expr),
        encoder_bias=_group_parameters("encoder_bias", old.joint_names_expr),
        delay_steps=P1_DELAY_STEPS,
    )
