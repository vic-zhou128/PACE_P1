"""Verify G1 reward provenance, P curricula and calibrated P physics."""

import argparse
import ast
from copy import deepcopy
import faulthandler
from importlib import import_module
import importlib.util
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
faulthandler.dump_traceback_later(120)
app = AppLauncher(args).app

import gymnasium as gym
import torch
import yaml
from isaaclab.managers import RewardTermCfg
from isaaclab.utils.types import ArticulationActions

import unitree_rl_lab.tasks
from unitree_rl_lab.assets.robots.p_pace_parameters import (
    JOINT_NAME_MAP, P1_DELAY_STEPS, P1_FITTED_DELAY, P1_MOTOR_CONFIG_SOURCE,
    P1_MOTOR_LIMITS, P1_MOTOR_LIMITS_BY_JOINT, P1_PARAMETERS, load_pace_parameters,
)
from unitree_rl_lab.assets.robots.pace_actuator import PaceDCMotor, PaceDCMotorCfg
from unitree_rl_lab.tasks.locomotion.robots.p_pace.velocity_env_cfg import PEnvCfg, PRewardsCfg
from unitree_rl_lab.tasks.locomotion.mdp.p_robot import balanced_omni_velocity_levels, push_robot_velocity_levels
from unitree_rl_lab.utils.export_deploy_cfg import export_deploy_cfg

G1 = import_module("unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg")


def check_actuator():
    """Use real Isaac Lab PD and DelayBuffer on CPU, with changing feedback."""
    for lag in (0, 1):
        cfg = PaceDCMotorCfg(
            joint_names_expr=["test"], stiffness=180.0, damping=10.0,
            saturation_effort=281.6, effort_limit=120.0, effort_limit_sim=120.0, velocity_limit=16.57,
            encoder_bias=0.1, delay_steps=lag,
        )
        motor = PaceDCMotor(cfg, joint_names=["test"], joint_ids=[0], num_envs=2, device="cpu")
        motor.reset(torch.arange(2))
        torques = []
        for position in (0.1, 0.2, 0.3):
            action = ArticulationActions(
                joint_positions=torch.zeros(2, 1), joint_velocities=torch.zeros(2, 1),
                joint_efforts=torch.zeros(2, 1),
            )
            result = motor.compute(action, torch.full((2, 1), position), torch.zeros(2, 1))
            torques.append(result.joint_efforts.clone())
            torch.testing.assert_close(motor.applied_effort, result.joint_efforts)
        expected = [0.0, -18.0, -36.0] if lag == 0 else [0.0, 0.0, -18.0]
        for torque, value in zip(torques, expected):
            torch.testing.assert_close(torque, torch.full((2, 1), value), atol=1e-5, rtol=1e-5)
        motor.reset(torch.tensor([0]))
        result = motor.compute(
            ArticulationActions(torch.zeros(2, 1), torch.zeros(2, 1), torch.zeros(2, 1)),
            torch.full((2, 1), 0.1), torch.zeros(2, 1),
        )
        torch.testing.assert_close(result.joint_efforts[0], torch.zeros(1))
    print("PASS: encoder bias sign, torque delay, applied-effort reporting and partial reset")


def check_identification_reference():
    """Check constants against actual fitting config without running that task."""
    path = Path(__file__).resolve().parents[2] / P1_MOTOR_CONFIG_SOURCE
    if not path.is_file():
        raise FileNotFoundError(f"Identification reference is missing: {path}")
    assignments = {
        node.targets[0].id: node.value for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    }
    fitted = ast.literal_eval(assignments["P1_FITTED_PARAMETERS"])
    assert ast.literal_eval(assignments["P1_FITTED_DELAY"]) == P1_FITTED_DELAY
    peak = ast.literal_eval(assignments["P1_SATURATION_EFFORTS"])
    curves = assignments["P1_PACE_ACTUATORS"]
    assert set(fitted) == set(JOINT_NAME_MAP)
    assert {ast.literal_eval(key) for key in curves.keys} == set(JOINT_NAME_MAP)
    for key, call in zip(curves.keys, curves.values):
        fit_name = ast.literal_eval(key)
        sim_name = JOINT_NAME_MAP[fit_name]
        assert ast.literal_eval(call.args[0]) == fit_name
        assert isinstance(call.args[1], ast.Subscript) and call.args[1].value.id == "P1_SATURATION_EFFORTS"
        assert ast.literal_eval(call.args[1].slice) == fit_name
        assert P1_MOTOR_LIMITS_BY_JOINT[sim_name] == {
            "saturation_effort": peak[fit_name],
            "effort_limit": ast.literal_eval(call.args[2]),
            "velocity_limit": ast.literal_eval(call.args[3]),
        }
        assert ast.literal_eval(call.args[4]) == 180.0 and ast.literal_eval(call.args[5]) == 10.0
        for parameter, value in zip(("armature", "viscous_friction", "friction", "encoder_bias"), fitted[fit_name]):
            assert P1_PARAMETERS[parameter][sim_name] == value
    print("PASS: all 12 joint fits, delays, gains and torque-speed limits match identification config")

    actuator_path = path.parents[3] / "utils" / "pace_actuator.py"
    spec = importlib.util.spec_from_file_location("pace_identification_reference", actuator_path)
    reference_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference_module)
    for group in ("hip_roll", "hip_yaw", "ankle_roll"):
        limits = P1_MOTOR_LIMITS[group]
        for lag in (0, 1):
            cfg = PaceDCMotorCfg(
                joint_names_expr=["test"], stiffness=180.0, damping=10.0,
                **limits, effort_limit_sim=limits["effort_limit"], encoder_bias=0.1, delay_steps=lag,
            )
            reference_cfg = cfg.copy()
            reference_cfg.max_delay = 5
            reference_cfg.initial_delay = lag
            kwargs = dict(joint_names=["test"], joint_ids=[0], num_envs=1, device="cpu")
            motor = PaceDCMotor(cfg, **kwargs)
            reference = reference_module.PaceDCMotor(reference_cfg, **kwargs)
            for speed in (0.0, 0.5, 0.95, 1.0, 1.2, 2.0, -0.5, -0.95, -1.0, -1.2, -2.0):
                for direction in (1.0, -1.0):
                    velocity = torch.tensor([[speed * limits["velocity_limit"]]])
                    position = torch.tensor([[0.2]])

                    def action():
                        return ArticulationActions(torch.tensor([[100.0 * direction]]), torch.zeros(1, 1),
                                                   torch.zeros(1, 1))

                    result = motor.compute(action(), position, velocity)
                    expected = reference.compute(action(), position, velocity)
                    torch.testing.assert_close(result.joint_efforts, expected.joint_efforts)
                    assert result.joint_efforts.abs().max() <= limits["effort_limit"] + 1e-5
                    if lag == 0 and speed == 0.95 and direction == 1.0:
                        torch.testing.assert_close(result.joint_efforts, torch.tensor([[0.05 * limits["saturation_effort"]]]),
                                                   atol=1e-4, rtol=1e-5)
                    if lag == 0 and speed == 1.0 and direction == 1.0:
                        torch.testing.assert_close(result.joint_efforts, torch.zeros(1, 1), atol=1e-4, rtol=1e-5)
    print("PASS: three motor curves match source PACE at stall, high speed, overspeed, reverse/braking and 0/1 delay")


def check_rewards_and_curricula():
    for path in (
        "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py",
        "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/rewards.py",
        "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/agents/rsl_rl_ppo_cfg.py",
    ):
        original = subprocess.check_output(["git", "show", f"HEAD:{path}"], cwd=PROJECT)
        assert (PROJECT / path).read_bytes() == original
    g1 = G1.RewardsCfg()
    p = PRewardsCfg()
    source_terms = {name: term for name, term in vars(g1).items() if isinstance(term, RewardTermCfg)}
    actual_terms = {name: term for name, term in vars(p).items() if isinstance(term, RewardTermCfg)}
    assert len(actual_terms) == 17
    assert set(actual_terms) == set(source_terms) - {"joint_deviation_arms", "joint_deviation_waists"}
    for name, term in actual_terms.items():
        assert term.func is source_terms[name].func
        assert term.weight == source_terms[name].weight
    assert p.base_height.params["target_height"] == 0.656
    assert p.gait.params["period"] == 0.8

    old = PROJECT.parent / "robot-p/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion"
    source = ast.parse((old / "mdp/curriculums.py").read_text())
    snapshot = ast.parse((PROJECT / "source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/mdp/p_robot/curriculums.py").read_text())
    for name in ("balanced_omni_velocity_levels", "push_robot_velocity_levels"):
        original = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == name)
        copied = next(n for n in snapshot.body if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(original) == ast.dump(copied)
    cfg = PEnvCfg()
    source_cfg = ast.parse((old / "robots/p/velocity_env_cfg.py").read_text())
    curriculum_class = next(n for n in source_cfg.body if isinstance(n, ast.ClassDef) and n.name == "CurriculumCfg")
    for n in curriculum_class.body:
        if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call):
            params = next(k.value for k in n.value.keywords if k.arg == "params")
            assert getattr(cfg.curriculum, n.targets[0].id).params == ast.literal_eval(params)
    print("PASS: official reward functions/weights and PPO defaults retained; original P curricula and settings match")

    # Exercise successful promotion, performance failure and push rollback with
    # completed-episode statistics at the real curriculum thresholds.
    count = 512
    ranges = deepcopy(cfg.commands.base_velocity.ranges)
    start = [tuple(getattr(ranges, n)) for n in ("lin_vel_x", "lin_vel_y", "ang_vel_z")]
    command = SimpleNamespace(
        cfg=SimpleNamespace(ranges=ranges, limit_ranges=cfg.commands.base_velocity.limit_ranges),
        axis_tracking_error_sum=torch.zeros(count, 3),
        axis_tracking_sample_count=torch.full((count, 3), 40.0),
        standing_motion_error_sum=torch.zeros(count),
        standing_sample_count=torch.full((count,), 20.0),
    )
    push_event = deepcopy(cfg.events.push_robot)
    env = SimpleNamespace(
        num_envs=count, device="cpu", common_step_counter=0, step_dt=0.02,
        episode_length_buf=torch.full((count,), 1000), max_episode_length=1000,
        command_manager=SimpleNamespace(get_term=lambda name: command),
        event_manager=SimpleNamespace(get_term_cfg=lambda name: push_event),
    )
    params = cfg.curriculum.velocity_levels.params
    push_params = cfg.curriculum.push_levels.params
    balanced_omni_velocity_levels(env, [], **params)
    env.common_step_counter = params["update_interval_steps"]
    outcome = balanced_omni_velocity_levels(env, list(range(count)), **params)
    assert outcome["promote_ready"].item() == 1.0
    assert all(outcome[f"direction_{axis}"].item() == 1.0 for axis in ("x", "y", "yaw"))
    push_robot_velocity_levels(env, [], **push_params)
    assert abs(env._push_velocity_curriculum_state["x_vel"] - 0.06) < 1e-6
    push_robot_velocity_levels(env, [], **push_params)
    assert abs(env._push_velocity_curriculum_state["x_vel"] - 0.06) < 1e-6
    command.axis_tracking_error_sum.fill_(10000.0)
    command.standing_motion_error_sum.fill_(10000.0)
    env.episode_length_buf.fill_(10)
    env.common_step_counter += params["update_interval_steps"]
    outcome = balanced_omni_velocity_levels(env, list(range(count)), **params)
    assert outcome["demote_ready"].item() == 1.0
    for name, original in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), start):
        assert tuple(getattr(ranges, name)) == original
    push_robot_velocity_levels(env, [], **push_params)
    assert env._push_velocity_curriculum_state["x_vel"] == push_params["start_x"]
    print("PASS: velocity promotion, rollback floor and synchronized push promotion/rollback")


def check_physics():
    cfg = PEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.commands.base_velocity.debug_vis = False
    cfg.seed = 7
    env = gym.make("Unitree-P-G1Rewards-PACE", cfg=cfg)
    try:
        env.reset()
        base = env.unwrapped
        robot = base.scene["robot"]
        assert set(robot.joint_names) == set(JOINT_NAME_MAP.values())
        assert len(base.reward_manager.active_terms) == 17
        assert set(base.curriculum_manager.active_terms) == {"velocity_levels", "push_levels"}
        actual_armature = robot.root_physx_view.get_dof_armatures().to(robot.device)
        actual_friction = robot.root_physx_view.get_dof_friction_properties().to(robot.device)
        for param, actual in (
            ("armature", actual_armature), ("friction", actual_friction[..., 0]),
            ("friction", actual_friction[..., 1]), ("viscous_friction", actual_friction[..., 2]),
        ):
            expected = torch.tensor([P1_PARAMETERS[param][name] for name in robot.joint_names], device=robot.device)
            torch.testing.assert_close(actual, expected.repeat(args.num_envs, 1))
        term = base.action_manager.get_term("JointPositionAction")
        bias = torch.tensor([P1_PARAMETERS["encoder_bias"][name] for name in robot.joint_names], device=robot.device)
        torch.testing.assert_close(term._offset + bias, robot.data.default_joint_pos)
        for actuator in robot.actuators.values():
            assert actuator.cfg.delay_steps == P1_DELAY_STEPS == 0
            assert actuator.cfg.stiffness == 180.0 and actuator.cfg.damping == 10.0
            for name in actuator.joint_names:
                limits = P1_MOTOR_LIMITS_BY_JOINT[name]
                assert actuator.cfg.saturation_effort == limits["saturation_effort"]
                assert actuator.cfg.effort_limit == actuator.cfg.effort_limit_sim == limits["effort_limit"]
                assert actuator.cfg.velocity_limit == limits["velocity_limit"]
        for _ in range(args.steps):
            obs, rewards, _, _, _ = env.step(torch.zeros(args.num_envs, 12, device=robot.device))
            assert obs["policy"].shape == (args.num_envs, 235)
            assert obs["critic"].shape == (args.num_envs, 250)
            assert torch.isfinite(obs["policy"]).all() and torch.isfinite(rewards).all()
        env.reset()
        output = Path("/tmp/unitree_p_g1_rewards_check")
        export_deploy_cfg(base, str(output))
        deploy = yaml.safe_load((output / "params/deploy.yaml").read_text())
        assert deploy["motor_identification"]["parameters_by_joint_name"] == P1_PARAMETERS
        assert deploy["motor_identification"]["motor_limits_by_joint_name"] == P1_MOTOR_LIMITS_BY_JOINT
        assert deploy["motor_identification"]["applied_delay_steps"] == 0
        print(f"PASS: 17 G1 rewards, 2 P curricula, PhysX fitted parameters, {args.steps} steps, reset and export")
    finally:
        env.close()


try:
    assert load_pace_parameters(PROJECT.parent / "p1_parameters_old_vs_new.csv")[0] == P1_PARAMETERS
    check_rewards_and_curricula()
    check_identification_reference()
    check_actuator()
    check_physics()
finally:
    faulthandler.cancel_dump_traceback_later()
    app.close()
