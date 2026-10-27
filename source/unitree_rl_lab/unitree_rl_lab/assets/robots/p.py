"""P robot asset configuration."""

from pathlib import Path

from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from .unitree import UnitreeArticulationCfg, UnitreeUrdfFileCfg


_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_P_URDF_PATH = _PROJECT_ROOT / "unitree_usd" / "p_urdf" / "urdf" / "p.urdf"


P_CFG = UnitreeArticulationCfg(
    spawn=UnitreeUrdfFileCfg(
        asset_path=str(_P_URDF_PATH),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.656),
        joint_pos={
            ".*hip_pitch.*": -0.2,
            ".*knee_pitch.*": 0.2,
            ".*ankle_pitch.*": -0.05,
            ".*hip_roll.*": 0.0,
            ".*hip_yaw.*": 0.0,
            ".*ankle_roll.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "hip_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*hip_roll.*"],
            effort_limit=120.0,
            effort_limit_sim=120.0,
            velocity_limit_sim=13.30,
            stiffness=180.0,
            damping=10.0,
            armature=0.06,
            min_delay=1,
            max_delay=1,
        ),
        "hip_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[".*hip_pitch.*"],
            effort_limit=120.0,
            effort_limit_sim=120.0,
            velocity_limit_sim=13.30,
            stiffness=180.0,
            damping=10.0,
            armature=0.06,
            min_delay=1,
            max_delay=1,
        ),
        "hip_yaw": DelayedPDActuatorCfg(
            joint_names_expr=[".*hip_yaw.*"],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=16.02,
            stiffness=180.0,
            damping=10.0,
            armature=0.0264,
            min_delay=1,
            max_delay=1,
        ),
        "knee": DelayedPDActuatorCfg(
            joint_names_expr=[".*knee_pitch.*"],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            velocity_limit_sim=16.02,
            stiffness=180.0,
            damping=10.0,
            armature=0.0264,
            min_delay=1,
            max_delay=1,
        ),
        "ankle_pitch": DelayedPDActuatorCfg(
            joint_names_expr=[".*ankle_pitch.*"],
            effort_limit=34.0,
            effort_limit_sim=34.0,
            velocity_limit_sim=8.69,
            stiffness=187.03872,
            damping=9.072,
            armature=0.03888,
            min_delay=1,
            max_delay=1,
        ),
        "ankle_roll": DelayedPDActuatorCfg(
            joint_names_expr=[".*ankle_roll.*"],
            effort_limit=34.0,
            effort_limit_sim=34.0,
            velocity_limit_sim=8.69,
            stiffness=187.03872,
            damping=9.072,
            armature=0.03888,
            min_delay=1,
            max_delay=1,
        ),
    },
    joint_sdk_names=[
        "hip_roll_l_joint",
        "hip_pitch_l_joint",
        "hip_yaw_l_joint",
        "knee_pitch_l_joint",
        "ankle_pitch_l_joint",
        "ankle_roll_l_joint",
        "hip_roll_r_joint",
        "hip_pitch_r_joint",
        "hip_yaw_r_joint",
        "knee_pitch_r_joint",
        "ankle_pitch_r_joint",
        "ankle_roll_r_joint",
    ],
)
