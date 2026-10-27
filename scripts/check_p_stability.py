"""Check standing anchors, stop trials and staged locomotion on CPU tensors."""

import argparse
from copy import deepcopy
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "unitree_rl_lab"))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import torch
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.curriculums import balanced_omni_velocity_levels
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.velocity_command import OmniVelocityCommand
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.walking_rewards import (
    standing_position_drift_l2, standing_heading_drift_l2,
)

TASK = "Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift"


def harness():
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    data = SimpleNamespace(
        root_pos_w=torch.zeros(4, 3), root_quat_w=torch.tensor([[1., 0, 0, 0]]).repeat(4, 1),
        root_lin_vel_b=torch.zeros(4, 3), root_ang_vel_b=torch.zeros(4, 3),
        projected_gravity_b=torch.tensor([[0., 0, -1.]]).repeat(4, 1),
    )

    class Scene:
        def __getitem__(self, name):
            return SimpleNamespace(data=data)

    env = SimpleNamespace(scene=Scene(), num_envs=4, device="cpu", step_dt=0.02,
                          common_step_counter=0, max_episode_length=1000,
                          episode_length_buf=torch.full((4,), 1000))
    cfg.commands.base_velocity.debug_vis = False
    command = OmniVelocityCommand(cfg.commands.base_velocity, env)
    env.command_manager = SimpleNamespace(get_term=lambda name: command)
    return cfg, env, command, data


def check_anchors_and_stops():
    _, env, command, data = harness()
    command.is_standing_env.zero_()
    command._update_standing_metrics()
    command.is_standing_env[:2] = True
    data.root_quat_w[:, 0] = torch.cos(torch.tensor(3.13 / 2))
    data.root_quat_w[:, 3] = torch.sin(torch.tensor(3.13 / 2))
    command._update_standing_metrics()
    assert command.stopping_attempts.tolist() == [1., 1., 0., 0.]
    data.root_pos_w[0, :2] = torch.tensor([0.03, 0.04])
    data.root_quat_w[0, 3] *= -1  # cross the +/- pi boundary
    data.root_pos_w[2, :2] = 1.0  # moving row must not get drift costs
    command._update_standing_metrics()
    torch.testing.assert_close(standing_position_drift_l2(env, "base_velocity"), torch.tensor([0.0025, 0, 0, 0]))
    heading = standing_heading_drift_l2(env, "base_velocity")
    assert abs(heading[0].item() - (2 * torch.pi - 6.26) ** 2) < 1e-6
    assert heading[1:].sum() == 0
    for _ in range(120):
        command._update_standing_metrics()
    assert command.stopping_attempts.tolist() == [1., 1., 0., 0.]
    assert command.stopping_successes.tolist() == [0., 1., 0., 0.]
    assert 0.48 <= command.stopping_time_sum[1] <= 0.54
    assert command.standing_component_sum[0, 2] > 0
    assert command.standing_component_sum[1, 2] == 0
    stable_before = command.standing_stable_samples[1].clone()
    metrics = command.reset(torch.tensor([0]))
    assert "standing_position_drift_m" in metrics
    assert command.standing_sample_count[0] == 0 and not command.standing_anchor_active[0]
    assert command.standing_stable_samples[1] == stable_before
    assert not command._stop_pending[0]
    print("PASS: fixed anchors, wrapped yaw, moving masks, stop success/timeout and partial reset", flush=True)


def check_stages():
    cfg, env, command, _ = harness()
    rewards = {name: deepcopy(getattr(cfg.rewards, name)) for name in
               ("gait", "feet_clearance", "feet_air_time", "swing_knee_flexion", "track_lin_vel_xy", "bootstrap_knee_lift")}
    original_weights = {k: v.weight for k, v in rewards.items()}
    physical_gait = SimpleNamespace(walking_time=torch.zeros(4), walking_totals=torch.zeros(4, 6))
    rewards["gait"].func = physical_gait
    env.reward_manager = SimpleNamespace(active_terms=list(rewards), get_term_cfg=lambda name: rewards[name],
                                         set_term_cfg=lambda name, value: rewards.__setitem__(name, value))
    events = {"push_robot": deepcopy(cfg.events.push_robot)}
    env.event_manager = SimpleNamespace(get_term_cfg=lambda name: events[name],
                                        set_term_cfg=lambda name, value: events.__setitem__(name, value))
    params = deepcopy(cfg.curriculum.velocity_levels.params)
    params.update(update_interval_steps=1, min_axis_samples=1, min_standing_samples=1,
                  min_completed_episodes=1, min_stopping_samples=1,
                  gait_min_moving_seconds=1., gait_min_transitions=1)
    balanced_omni_velocity_levels(env, [], **params)
    assert rewards["gait"].weight == original_weights["gait"] * 0.25
    assert command.cfg.rel_standing_envs == 0.5
    assert events["push_robot"].params["velocity_range"]["x"] == (-0.05, 0.05)
    assert rewards["track_lin_vel_xy"].params["gait_exploration_fraction"] == 1.0

    def window(good=True, gait_case="good"):
        env.common_step_counter += 1
        command.axis_tracking_sample_count.zero_()
        probs = command.cfg.mode_probabilities
        for axis in range(3):
            if probs[axis] > 0 or probs[3] > 0:
                command.axis_tracking_sample_count[:, axis] = 10
        command.axis_tracking_error_sum.zero_()
        command.standing_sample_count.fill_(10)
        command.standing_motion_error_sum.fill_(0 if good else 10)
        command.standing_stable_samples.fill_(10 if good else 0)
        command.stopping_attempts.fill_(1)
        command.stopping_successes.fill_(1 if good else 0)
        physical_gait.walking_time.fill_(10.)
        physical_gait.walking_totals[:] = torch.tensor([10., 10., 0.6, 0.6, 18., 18.])
        if gait_case == "none":
            physical_gait.walking_totals.zero_()
        elif gait_case == "sparse":
            physical_gait.walking_totals[:] = torch.tensor([1., 1., 0.06, 0.06, 1., 1.])
        elif gait_case == "low":
            physical_gait.walking_totals[:, 2:4] = 0.21
        elif gait_case == "repeat":
            physical_gait.walking_totals[:, 5] = 0
        return balanced_omni_velocity_levels(env, list(range(4)), **params)

    for case in ("none", "sparse", "low", "repeat"):
        outcome = window(gait_case=case)
        assert outcome["stage"] == 0 and outcome["gait_score"] < 1
        if case in ("none", "sparse"):
            assert outcome["bootstrap_progress"] == 0
            assert outcome["velocity_reward_floor"] == 1
    assert window()["stage"] == 0  # cannot advance on one good window
    assert window()["stage"] == 1  # y/yaw were deliberately unsampled in stage 0
    floor = rewards["track_lin_vel_xy"].params["gait_exploration_fraction"]
    assert 0.25 < floor < 1.0
    assert 0 < rewards["bootstrap_knee_lift"].weight < original_weights["bootstrap_knee_lift"]
    assert command.cfg.mode_probabilities[2] == 0
    assert rewards["feet_clearance"].weight == original_weights["feet_clearance"] * 0.5
    assert window()["stage"] == 1
    assert window()["stage"] == 2  # yaw still unsampled in stage 1
    assert command.cfg.rel_standing_envs == 0.25
    assert command.cfg.mode_probabilities[2] > 0
    assert rewards["gait"].weight == original_weights["gait"]
    assert window(False)["stage"] == 2
    assert window(False)["stage"] == 1
    assert window(False)["stage"] == 1
    assert window(False)["stage"] == 0
    assert window(False)["stage"] == 0
    assert rewards["gait"].weight == original_weights["gait"] * 0.25
    assert rewards["track_lin_vel_xy"].params["gait_exploration_fraction"] <= floor
    # Standing alone is insufficient: no measured walking-to-stop attempts.
    state = env._balanced_omni_curriculum_state
    command.stopping_attempts.zero_()
    command.stopping_successes.zero_()
    state["stopping_attempts"] = 0
    env.common_step_counter += 1
    balanced_omni_velocity_levels(env, list(range(4)), **params)
    assert state["last_update_step"] < env.common_step_counter
    print("PASS: physical step-rate/height/alternation gates reject shuffling; staged sampling, stop gate and reward scales", flush=True)


try:
    check_anchors_and_stops()
    check_stages()
    print("PASS: all stability and physical-gait curriculum checks completed", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
app.close()
