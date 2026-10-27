"""P1 PACE locomotion environment configuration.

Current task: Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth.
Edit the complete current TrueLiftSmooth configuration in this file:
RobotSceneCfg, EventCfg, CommandsCfg, ActionsCfg, ObservationsCfg, RewardsCfg,
TerminationsCfg, CurriculumCfg, RobotEnvCfg and RobotPlayEnvCfg.

Historical task variants are collected at the end for existing checkpoints.
Robot geometry/actuator definitions live in assets/robots; MDP functions live
in locomotion/mdp, just as in the G1 environment.
"""

import math
import os
import xml.etree.ElementTree as ET

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab.utils.version import get_isaac_sim_version

from unitree_rl_lab.assets.robots.p_pace import P_PACE_CFG
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.mdp import p_robot
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.foot_trajectory_rewards import (
    AlternatingFootstepReward,
    alternating_air_time_credit,
    alternating_knee_lift_credit,
    alternating_sole_lift_credit,
    bootstrap_knee_lift_credit,
    landing_knee_excess,
    swing_hip_roll_excess,
    swing_sole_clearance,
)


@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """P1 robot, flat ground, foot contact sensor and lighting."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = P_PACE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    # Both ankles: pitch and roll position/velocity feedback gains.
    robot.actuators["ankle_pitch"].stiffness = 40.0
    robot.actuators["ankle_pitch"].damping = 2.0
    robot.actuators["ankle_roll"].stiffness = 40.0
    robot.actuators["ankle_roll"].damping = 2.0
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class EventCfg:
    """G1 randomization, resets and pushes adapted to P1's pelvis."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (-1.0, 1.0),
        },
    )

    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 5.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class CommandsCfg:
    """Walking command sampling, speed limits and acceleration limits."""

    base_velocity = p_robot.OmniVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 7.0),
        rel_standing_envs=0.50,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        mode_probabilities=(1.0, 0.0, 0.0, 0.0),
        min_command=(0.07, 0.05, 0.08),
        acceleration=(0.80, 0.60, 1.20),
        deceleration=(1.20, 0.90, 1.80),
        ranges=p_robot.OmniVelocityCommandCfg.Ranges(
            lin_vel_x=(0.08, 0.16),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(0.0, 0.0),
        ),
        limit_ranges=p_robot.OmniVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.55),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.60, 0.60),
        ),
    )


@configclass
class ActionsCfg:
    """Joint targets, per-joint action scales, clipping and command delays."""

    JointPositionAction = p_robot.PaceJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale={
            ".*hip_roll.*": 0.16,
            ".*hip_yaw.*": 0.10,
            ".*hip_pitch.*": 0.32,
            ".*knee_pitch.*": 0.36,
            ".*ankle_pitch.*": 0.18,
            ".*ankle_roll.*": 0.10,
        },
        use_default_offset=True,
        raw_action_clip=1.0,
        # Policy runs at 50 Hz; PACE actuation runs at 200 Hz with zero delay.
        min_inference_delay=0,
        max_inference_delay=0,
        min_communication_delay=0,
        max_communication_delay=0,
        clip={
            ".*hip_roll.*": (-0.22, 0.22),
            ".*hip_yaw.*": (-0.16, 0.16),
            ".*hip_pitch.*": (-0.28, 0.35),
            ".*knee_pitch.*": (-0.22, 0.38),
            ".*ankle_pitch.*": (-0.14, 0.20),
            ".*ankle_roll.*": (-0.12, 0.12),
        },
    )


@configclass
class ObservationsCfg:
    """No phase input: actor 45 x 15 frames; critic 48 x 15 frames."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05, noise=Unoise(n_min=-1.5, n_max=1.5))
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 15
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel, scale=0.05)
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.history_length = 15

    # privileged observations
    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Complete TrueLiftSmooth rewards; tune weights and parameters here."""

    # -- task
    track_lin_vel_xy = RewTerm(
        func=p_robot.track_lin_vel_xy_progress,
        weight=2.0,
        params={
            "command_name": "base_velocity", "std": 0.20,
            "gait_reward_name": "gait", "gait_exploration_fraction": 1.0,
        },
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp, weight=0.5, params={"command_name": "base_velocity", "std": math.sqrt(0.25)}
    )

    alive = RewTerm(func=mdp.is_alive, weight=0.15)

    # -- base
    base_linear_velocity = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
    base_angular_velocity = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.10)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-0.001)
    joint_acc = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.03)
    # Extra costs apply only to settled zero commands; walking keeps its own costs.
    standing_base_xy = RewTerm(
        func=p_robot.standing_base_xy_l2, weight=-4.0, params={"command_name": "base_velocity"}
    )
    standing_base_yaw = RewTerm(
        func=p_robot.standing_base_yaw_l2, weight=-1.0, params={"command_name": "base_velocity"}
    )
    standing_joint_vel = RewTerm(
        func=p_robot.standing_joint_vel_l2, weight=-0.02, params={"command_name": "base_velocity"}
    )
    standing_action_rate = RewTerm(
        func=p_robot.standing_action_rate_l2, weight=-0.10, params={"command_name": "base_velocity"}
    )
    standing_position_drift = RewTerm(
        func=p_robot.standing_position_drift_l2, weight=-8.0, params={"command_name": "base_velocity"}
    )
    standing_heading_drift = RewTerm(
        func=p_robot.standing_heading_drift_l2, weight=-2.0, params={"command_name": "base_velocity"}
    )
    standing_contact_spin = RewTerm(
        func=p_robot.standing_contact_spin_l2,
        weight=-0.20,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-5.0)
    energy = RewTerm(func=mdp.energy, weight=-2e-5)

    joint_deviation_arms = None
    joint_deviation_waists = None
    joint_deviation_legs = RewTerm(
        func=mdp.joint_deviation_l1,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*hip_roll.*", ".*hip_yaw.*"])},
    )

    # -- robot
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    base_height = RewTerm(func=mdp.base_height_l2, weight=-10, params={"target_height": 0.656})

    # -- feet
    gait = RewTerm(
        func=AlternatingFootstepReward,
        weight=0.5,
        params={
            "command_name": "base_velocity",
            "takeoff_clearance": 0.01,
            "landing_clearance": 0.005,
            "min_step_height": 0.02,
            "peak_clearance": 0.06,
            "quality_window_s": 4.0,
            "quality_foot_rate_hz": 0.4,
            "quality_height_m": 0.025,
            "height_floor": 0.002,
            "min_swing_time": 0.12,
            "max_swing_time": 0.6,
            "landing_time": 0.06,
            "min_step_interval": 0.25,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.2,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_clearance = RewTerm(
        func=alternating_sole_lift_credit,
        weight=0.5,
        params={"gait_reward_name": "gait"},
    )

    # -- other
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1,
        params={
            "threshold": 1,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["(?!.*ankle.*).*"]),
        },
    )

    feet_air_time = RewTerm(
        func=alternating_air_time_credit,
        weight=0.5,
        params={"gait_reward_name": "gait"},
    )

    air_time_variance = RewTerm(
        func=mdp.air_time_variance_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            )
        },
    )


@configclass
class TerminationsCfg:
    """Episode timeout, low pelvis, excessive tilt and pelvis contact."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.42})
    bad_orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.65})
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names="pelvis_link"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Stages gate standing, continuous walking, stopping and push difficulty."""

    velocity_levels = CurrTerm(
        func=p_robot.balanced_omni_velocity_levels,
        params={
            "update_interval_steps": 10000,
            "range_steps": (0.03, 0.02, 0.04),
            "tracking_stds": (0.15, 0.12, 0.18),
            "promote_tracking_score": 0.75,
            "demote_tracking_score": 0.45,
            "promote_survival_rate": 0.85,
            "demote_survival_rate": 0.70,
            "min_standing_samples": 10000,
            "promote_standing_score": 0.72,
            "demote_standing_score": 0.45,
            # Reward scales multiply the task's nominal gait/lift/air/knee weights.
            # Stages advance/retreat only after two consecutive evaluation windows.
            "stages": (
                {
                    "standing_fraction": 0.50,
                    "mode_probabilities": (1.0, 0.0, 0.0, 0.0),
                    "ranges": ((0.08, 0.16), (0.0, 0.0), (0.0, 0.0)),
                    "lift_reward_scale": 0.25,
                    "push_velocity": (0.05, 0.03),
                    "gait_min_foot_rate_hz": 0.4,
                    "gait_min_height_m": 0.025,
                },
                {
                    "standing_fraction": 0.35,
                    "mode_probabilities": (0.85, 0.15, 0.0, 0.0),
                    "ranges": ((0.08, 0.24), (-0.06, 0.06), (0.0, 0.0)),
                    "lift_reward_scale": 0.50,
                    "push_velocity": (0.15, 0.10),
                    "gait_min_foot_rate_hz": 0.5,
                    "gait_min_height_m": 0.04,
                },
                {
                    "standing_fraction": 0.25,
                    "mode_probabilities": (0.70, 0.10, 0.10, 0.10),
                    "ranges": ((-0.12, 0.30), (-0.08, 0.08), (-0.15, 0.15)),
                    "lift_reward_scale": 1.0,
                    "push_velocity": (0.50, 0.50),
                    "gait_min_foot_rate_hz": 0.6,
                    "gait_min_height_m": 0.05,
                },
            ),
            "stage_push_term_name": "push_robot",
            "stage_promote_windows": 2,
            "stage_demote_windows": 2,
            "min_stopping_samples": 50,
            "promote_stable_fraction": 0.80,
            "promote_stopping_success": 0.80,
            "physical_gait_term_name": "gait",
            "gait_min_moving_seconds": 200.0,
            "gait_min_transitions": 100,
            "gait_min_alternation": 0.90,
            "bootstrap_schedule": True,
            "bootstrap_velocity_floor_start": 1.0,
            "bootstrap_velocity_floor_end": 0.25,
            "bootstrap_adaptation_rate": 0.25,
        },
    )
    push_levels = None


@configclass
class RobotEnvCfg(ManagerBasedRLEnvCfg):
    """Current training environment, built directly from Isaac Lab's base."""

    scene: RobotSceneCfg = RobotSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # Same Isaac Lab timing as official G1 and robot-p. The P plane scene
        # has no torso height scanner or generated-terrain curriculum.
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.sky_light.spawn.texture_file = None
        path = os.environ.get("P_ROBOT_URDF")
        if path:
            self.scene.robot.spawn.asset_path = os.path.abspath(os.path.expanduser(path))
        path = self.scene.robot.spawn.asset_path
        if not os.path.isfile(path):
            raise FileNotFoundError(f"P URDF is missing: {path}")
        joints = {
            j.attrib["name"] for j in ET.parse(path).getroot().findall("joint") if j.attrib.get("type") != "fixed"
        }
        if joints != set(self.scene.robot.joint_sdk_names):
            raise ValueError("P URDF joints differ from PACE parameter mapping")
        if get_isaac_sim_version().major < 5:
            raise RuntimeError("PACE dynamic/viscous joint friction requires Isaac Sim >= 5.0")


@configclass
class RobotPlayEnvCfg(RobotEnvCfg):
    """Single-robot playback with deterministic commands and reset settings."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.0
        command.mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        command.ranges.lin_vel_x = (0.25, 0.25)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        self.events.reset_base.params["pose_range"] = {
            name: (0.0, 0.0) for name in self.events.reset_base.params["pose_range"]
        }
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.curriculum.velocity_levels = None
        self.curriculum.push_levels = None


@configclass
class KneeLiftActionsCfg(ActionsCfg):
    """Extra knee flexion travel for the new task, preserving old policy scales."""

    def __post_init__(self):
        self.JointPositionAction.scale[".*knee_pitch.*"] = 0.55
        self.JointPositionAction.clip[".*knee_pitch.*"] = (-0.22, 0.55)


@configclass
class KneeLiftRewardsCfg(RewardsCfg):
    """Soft knee/hip coordination constraints on the existing alternating gait."""

    swing_knee_flexion = RewTerm(
        func=alternating_knee_lift_credit, weight=1.0, params={"gait_reward_name": "gait"}
    )
    bootstrap_knee_lift = RewTerm(
        func=bootstrap_knee_lift_credit, weight=0.5, params={"gait_reward_name": "gait"}
    )
    swing_hip_roll_excess = RewTerm(
        func=swing_hip_roll_excess, weight=-0.5, params={"gait_reward_name": "gait"}
    )
    landing_knee_excess = RewTerm(
        func=landing_knee_excess, weight=-0.15, params={"gait_reward_name": "gait"}
    )

    def __post_init__(self):
        # Prefer actual alternating lift over the previous near-ground gait.
        # These are event-budget weights, not rewards for holding a leg up.
        self.gait.weight = 1.5
        self.feet_clearance.weight = 2.0
        self.gait.params.update(
            bootstrap_enabled=True,
            bootstrap_height_target=0.025,
            bootstrap_min_height=0.006,
            bootstrap_knee_start=0.01,
            bootstrap_knee_target=0.18,
            bootstrap_max_time=0.8,
            bootstrap_min_unload_time=0.06,
            knee_joint_cfg=SceneEntityCfg(
                "robot", joint_names=["left_knee_pitch_joint", "right_knee_pitch_link"], preserve_order=True
            ),
            hip_roll_joint_cfg=SceneEntityCfg(
                "robot", joint_names=["left_hip_roll_link", "right_hip_roll_link"], preserve_order=True
            ),
            knee_flexion_start=0.08,
            knee_flexion_target=0.42,
            knee_separation_start=0.04,
            knee_separation_target=0.26,
            hip_roll_allowance=0.08,
            hip_roll_scale=0.10,
            lateral_roll_allowance=0.08,
            yaw_roll_allowance=0.04,
            landing_knee_allowance=0.16,
            landing_knee_scale=0.20,
        )


@configclass
class KneeLiftEnvCfg(RobotEnvCfg):
    actions: KneeLiftActionsCfg = KneeLiftActionsCfg()
    rewards: KneeLiftRewardsCfg = KneeLiftRewardsCfg()


@configclass
class KneeLiftPlayEnvCfg(RobotPlayEnvCfg):
    actions: KneeLiftActionsCfg = KneeLiftActionsCfg()
    rewards: KneeLiftRewardsCfg = KneeLiftRewardsCfg()


# -----------------------------------------------------------------------------
# Historical task variants. Tune TrueLiftSmooth/KneeLift rewards above.
# Each registered environment uses RobotEnvCfg directly; no cross-file chain.
# -----------------------------------------------------------------------------

LEGACY_GAIT_PERIOD = 0.8


@configclass
class PCommandsCfg:
    base_velocity = p_robot.OmniVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 7.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=True,
        mode_probabilities=(0.30, 0.25, 0.20, 0.25),
        min_command=(0.07, 0.05, 0.08),
        acceleration=(0.80, 0.60, 1.20),
        deceleration=(1.20, 0.90, 1.80),
        ranges=p_robot.OmniVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.12, 0.18),
            lin_vel_y=(-0.08, 0.08),
            ang_vel_z=(-0.15, 0.15),
        ),
        limit_ranges=p_robot.OmniVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.55),
            lin_vel_y=(-0.30, 0.30),
            ang_vel_z=(-0.60, 0.60),
        ),
    )


@configclass
class PEventsCfg:
    """Moderate domain randomization aimed at transfer, not task destruction."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            # Keep dynamic friction no greater than static friction for every sample.
            "static_friction_range": (0.85, 1.15),
            "dynamic_friction_range": (0.60, 0.85),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "mass_distribution_params": (-0.5, 1.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="pelvis_link"),
            "com_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.005, 0.005)},
        },
    )
    actuator_gains = None
    joint_parameters = None

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.02, 0.02),
                "y": (-0.02, 0.02),
                "roll": (-0.03, 0.03),
                "pitch": (-0.03, 0.03),
                "yaw": (-0.05, 0.05),
            },
            "velocity_range": {
                "x": (-0.08, 0.08),
                "y": (-0.08, 0.08),
                "z": (-0.03, 0.03),
                "roll": (-0.08, 0.08),
                "pitch": (-0.08, 0.08),
                "yaw": (-0.08, 0.08),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.04, 0.04), "velocity_range": (-0.2, 0.2)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(12.0, 18.0),
        params={
            "velocity_range": {
                "x": (-0.04, 0.04),
                "y": (-0.03, 0.03),
                "yaw": (-0.06, 0.06),
            }
        },
    )


@configclass
class PCurriculumCfg:
    velocity_levels = CurrTerm(
        func=p_robot.balanced_omni_velocity_levels,
        params={
            "update_interval_steps": 10000,
            "range_steps": (0.03, 0.02, 0.04),
            "tracking_stds": (0.15, 0.12, 0.18),
            "promote_tracking_score": 0.75,
            "demote_tracking_score": 0.45,
            "promote_standing_score": 0.72,
            "demote_standing_score": 0.45,
            "promote_survival_rate": 0.96,
            "demote_survival_rate": 0.82,
        },
    )
    push_levels = CurrTerm(
        func=p_robot.push_robot_velocity_levels,
        params={
            "event_term_name": "push_robot",
            "start_x": 0.04,
            "start_y": 0.03,
            "start_yaw": 0.06,
            "end_x": 0.18,
            "end_y": 0.12,
            "end_yaw": 0.18,
            "step_x": 0.02,
            "step_y": 0.01,
            "step_yaw": 0.02,
        },
    )


@configclass
class WalkCurriculumCfg(PCurriculumCfg):
    def __post_init__(self):
        self.velocity_levels.params.update(
            min_standing_samples=0,
            promote_standing_score=0.0,
            demote_standing_score=-1.0,
            promote_survival_rate=0.85,
            demote_survival_rate=0.70,
        )


@configclass
class PObservationsCfg(ObservationsCfg):
    """Official history length 5, plus the requested gait phase: 47 x 5."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": LEGACY_GAIT_PERIOD})

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    @configclass
    class CriticCfg(ObservationsCfg.CriticCfg):
        gait_phase = ObsTerm(func=mdp.gait_phase, params={"period": LEGACY_GAIT_PERIOD})

        def __post_init__(self):
            super().__post_init__()
            self.history_length = 5

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class PRewardsCfg(RewardsCfg):
    """Original G1 rewards adapted to P morphology."""

    feet_clearance = RewTerm(
        func=mdp.foot_clearance_reward,
        weight=1.0,
        params={
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg(
                "robot", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        },
    )
    feet_air_time = None
    air_time_variance = None

    def __post_init__(self):
        self.track_lin_vel_xy.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.track_lin_vel_xy.weight = 1.0
        self.track_lin_vel_xy.params["std"] = math.sqrt(0.25)
        self.base_angular_velocity.weight = -0.05
        self.action_rate.weight = -0.05
        self.gait.func = mdp.feet_gait
        self.gait.params = {
            "period": LEGACY_GAIT_PERIOD,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        }


@configclass
class WalkRewardsCfg(PRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        self.track_lin_vel_xy.weight = 2.0
        self.track_lin_vel_xy.params["std"] = 0.20
        self.feet_clearance.func = p_robot.foot_clearance_during_swing
        self.feet_clearance.params.update(
            sensor_cfg=SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            period=LEGACY_GAIT_PERIOD,
            offset=(0.0, 0.5),
            threshold=0.55,
            command_name="base_velocity",
        )


@configclass
class NoPhase15WalkRewardsCfg(RewardsCfg):
    feet_air_time = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "threshold": 0.4,
        },
    )

    def __post_init__(self):
        self.track_lin_vel_xy.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.base_angular_velocity.weight = -0.05
        self.action_rate.weight = -0.05
        self.feet_clearance.weight = 1.0
        self.gait.func = mdp.feet_gait
        self.gait.params = {
            "period": 1.0,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True
            ),
        }
        self.feet_clearance.func = swing_sole_clearance
        self.feet_clearance.params = {
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
                preserve_order=True,
            ),
            "command_name": "base_velocity",
            "period": 1.0,
            "offset": (0.0, 0.5),
            "threshold": 0.55,
            "peak_clearance": 0.06,
            "height_std": 0.02,
        }


@configclass
class DenseLiftRewardsCfg(NoPhase15WalkRewardsCfg):
    def __post_init__(self):
        super().__post_init__()
        # The old 2 cm Gaussian is nearly zero for a foot lifted only 1-2 cm.
        # Keep the 6 cm goal, but make intermediate lifts useful to PPO.
        self.feet_clearance.params["height_std"] = 0.04
        self.feet_clearance.weight = 2.0


@configclass
class PEnvCfg(RobotEnvCfg):
    observations: PObservationsCfg = PObservationsCfg()
    commands: PCommandsCfg = PCommandsCfg()
    rewards: PRewardsCfg = PRewardsCfg()
    events: PEventsCfg = PEventsCfg()
    curriculum: PCurriculumCfg = PCurriculumCfg()


@configclass
class PPlayEnvCfg(PEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        self.commands.base_velocity.ranges = self.commands.base_velocity.limit_ranges
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.curriculum.velocity_levels = None
        self.curriculum.push_levels = None


@configclass
class WalkEnvCfg(RobotEnvCfg):
    observations: PObservationsCfg = PObservationsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()
    events: PEventsCfg = PEventsCfg()
    curriculum: WalkCurriculumCfg = WalkCurriculumCfg()


@configclass
class WalkPlayEnvCfg(WalkEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.episode_length_s = 60.0
        command = self.commands.base_velocity
        command.rel_standing_envs = 0.0
        command.mode_probabilities = (1.0, 0.0, 0.0, 0.0)
        command.ranges.lin_vel_x = (0.25, 0.25)
        command.ranges.lin_vel_y = (0.0, 0.0)
        command.ranges.ang_vel_z = (0.0, 0.0)
        self.observations.policy.enable_corruption = False
        self.events.physics_material = None
        self.events.add_base_mass = None
        self.events.base_com = None
        self.events.push_robot = None
        self.curriculum.velocity_levels = None
        self.curriculum.push_levels = None


@configclass
class G1WalkEnvCfg(RobotEnvCfg):
    observations: PObservationsCfg = PObservationsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()


@configclass
class G1WalkPlayEnvCfg(RobotPlayEnvCfg):
    observations: PObservationsCfg = PObservationsCfg()
    rewards: WalkRewardsCfg = WalkRewardsCfg()


@configclass
class NoPhase15WalkEnvCfg(RobotEnvCfg):
    rewards: NoPhase15WalkRewardsCfg = NoPhase15WalkRewardsCfg()


@configclass
class NoPhase15WalkPlayEnvCfg(RobotPlayEnvCfg):
    rewards: NoPhase15WalkRewardsCfg = NoPhase15WalkRewardsCfg()


@configclass
class DenseLiftEnvCfg(RobotEnvCfg):
    rewards: DenseLiftRewardsCfg = DenseLiftRewardsCfg()


@configclass
class DenseLiftPlayEnvCfg(RobotPlayEnvCfg):
    rewards: DenseLiftRewardsCfg = DenseLiftRewardsCfg()


# Explicit names retained for users importing the completed experiments.
TrueLiftSmoothRewardsCfg = RewardsCfg
TrueLiftSmoothEnvCfg = RobotEnvCfg
TrueLiftSmoothPlayEnvCfg = RobotPlayEnvCfg
