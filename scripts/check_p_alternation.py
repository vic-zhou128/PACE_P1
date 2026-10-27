"""Check physical step ordering, shared rewards, resets and bounded PPO noise."""

import argparse
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
from tensordict import TensorDict
from isaaclab.managers import RewardTermCfg, SceneEntityCfg
from isaaclab_rl.rsl_rl import export_policy_as_jit
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.tasks.locomotion.mdp.p_robot.foot_trajectory_rewards import (
    AlternatingFootstepReward,
    alternating_air_time_credit,
    alternating_knee_lift_credit,
    alternating_sole_lift_credit,
    bootstrap_knee_lift_credit,
    landing_knee_excess,
    swing_hip_roll_excess,
)
from unitree_rl_lab.utils.rsl_rl_runner import make_on_policy_runner, initialize_policy_for_finetuning
from unitree_rl_lab.tasks.locomotion.mdp.p_robot import walking_rewards
from unitree_rl_lab.utils.p_playback import keyboard_speed_ranges


def check_keyboard_limits():
    limits = SimpleNamespace(lin_vel_x=(-0.45, 0.55), lin_vel_y=(-0.30, 0.30))
    forward_only = {"lin_vel_x": (0.08, 0.16), "lin_vel_y": (0.0, 0.0)}
    result = keyboard_speed_ranges(limits, 0.15, 0.05, forward_only)
    assert result.lin_vel_x == (0.0, 0.15) and result.lin_vel_y == (0.0, 0.0)
    assert limits.lin_vel_x == (-0.45, 0.55)
    mixed = {"lin_vel_x": (-0.12, 0.30), "lin_vel_y": (-0.08, 0.08)}
    result = keyboard_speed_ranges(limits, 0.15, 0.05, mixed)
    assert result.lin_vel_x == (-0.12, 0.15) and result.lin_vel_y == (-0.05, 0.05)
    result = keyboard_speed_ranges(limits, 0.9, 0.9, mixed)
    assert result.lin_vel_x == (-0.12, 0.30) and result.lin_vel_y == (-0.08, 0.08)
    try:
        keyboard_speed_ranges(limits, float("nan"), 0.05)
    except ValueError:
        pass
    else:
        raise AssertionError("Nonfinite keyboard speed was accepted")
    print("PASS: keyboard speeds respect checkpoint training ranges and disable unsampled directions", flush=True)


def check_standing_costs():
    # Rows: standing, slow forward ramp, yaw-only command, braking into stand.
    command = SimpleNamespace(
        command=torch.tensor([[0., 0., 0.], [0.01, 0., 0.], [0., 0., 0.2], [0.1, 0., 0.]]),
        is_standing_env=torch.tensor([True, False, False, True]),
        cfg=SimpleNamespace(standing_command_threshold=0.04),
    )
    data = SimpleNamespace(
        root_lin_vel_b=torch.tensor([[0.3, 0.4, 0.]]).repeat(4, 1),
        root_ang_vel_b=torch.tensor([[0., 0., 2.]]).repeat(4, 1),
        joint_vel=torch.ones(4, 12),
        body_ang_vel_w=torch.tensor([[[0., 0., 2.], [0., 0., 3.]]]).repeat(4, 1, 1),
    )
    force = torch.zeros(4, 3, 2, 3)
    force[:, 0, 0, 2] = 10.  # left contact in history; right airborne
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w_history=force))

    class Scene:
        sensors = {"contact_forces": sensor}

        def __getitem__(self, name):
            return SimpleNamespace(data=data)

    env = SimpleNamespace(
        scene=Scene(), command_manager=SimpleNamespace(get_term=lambda name: command),
        action_manager=SimpleNamespace(action=torch.ones(4, 12), prev_action=torch.zeros(4, 12)),
    )
    feet = SceneEntityCfg("robot"); feet.body_ids = [0, 1]
    contact = SceneEntityCfg("contact_forces"); contact.body_ids = [0, 1]
    for func, expected in (
        (walking_rewards.standing_base_xy_l2, 0.25),
        (walking_rewards.standing_base_yaw_l2, 4.),
        (walking_rewards.standing_joint_vel_l2, 12.),
        (walking_rewards.standing_action_rate_l2, 12.),
    ):
        torch.testing.assert_close(func(env, "base_velocity"), torch.tensor([expected, 0., 0., 0.]))
    spin = lambda: walking_rewards.standing_contact_spin_l2(env, "base_velocity", feet, contact)
    torch.testing.assert_close(spin(), torch.tensor([4., 0., 0., 0.]))
    force.zero_()
    assert spin().sum() == 0
    data.root_lin_vel_b.zero_(); data.root_ang_vel_b.zero_(); data.joint_vel.zero_()
    env.action_manager.prev_action.copy_(env.action_manager.action)
    for func in (walking_rewards.standing_base_xy_l2, walking_rewards.standing_base_yaw_l2,
                 walking_rewards.standing_joint_vel_l2, walking_rewards.standing_action_rate_l2):
        assert func(env, "base_velocity").sum() == 0
    cfg = load_cfg_from_registry(
        "Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift", "env_cfg_entry_point"
    )
    assert cfg.commands.base_velocity.rel_standing_envs == 0.50
    gate = cfg.curriculum.velocity_levels.params
    assert gate["min_standing_samples"] == 10000
    assert gate["promote_standing_score"] == 0.72 and gate["demote_standing_score"] == 0.45
    print("PASS: standing-only costs, moving ramps, yaw commands, braking, contact spin and curriculum gate", flush=True)


class StepHarness:
    def __init__(self, count=2, knee_shaping=False, bootstrap=False):
        self.height = torch.zeros(count, 2)
        self.position = torch.zeros(count, 2, 3)
        self.quaternion = torch.tensor([1.0, 0, 0, 0]).expand(count, 2, 4).clone()
        self.contacts = torch.ones(count, 2)
        self.commands = torch.tensor([[0.25, 0, 0]]).expand(count, 3).clone()
        self.joint_pos = torch.zeros(count, 4)
        self.knee_shaping = knee_shaping
        self.bootstrap = bootstrap
        self.forces = torch.zeros(count, 1, 2, 3)
        robot = SimpleNamespace(data=SimpleNamespace(
            body_pos_w=self.position, body_quat_w=self.quaternion,
            joint_pos=self.joint_pos, default_joint_pos=torch.zeros_like(self.joint_pos),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).expand(count, 3),
            root_quat_w=torch.tensor([[1.0, 0, 0, 0]]).expand(count, 4),
            root_lin_vel_w=torch.tensor([[0.25, 0, 0]]).expand(count, 3).clone(),
        ))
        sensor = SimpleNamespace(data=SimpleNamespace(current_contact_time=self.contacts, net_forces_w_history=self.forces))

        class Scene:
            env_origins = torch.zeros(count, 3)
            sensors = {"contact_forces": sensor}

            def __getitem__(self, name):
                return robot

        self.env = SimpleNamespace(
            scene=Scene(), num_envs=count, device="cpu", step_dt=0.02,
            common_step_counter=0, extras={"log": {}},
            command_manager=SimpleNamespace(get_command=lambda name: self.commands),
        )
        asset = SceneEntityCfg("robot"); asset.body_ids = [0, 1]
        contact = SceneEntityCfg("contact_forces"); contact.body_ids = [0, 1]
        self.cfg = RewardTermCfg(
            func=AlternatingFootstepReward, weight=0.5,
            params={"asset_cfg": asset, "sensor_cfg": contact, "command_name": "base_velocity"},
        )
        if knee_shaping:
            knee = SceneEntityCfg("robot")
            knee.joint_ids = [0, 1]
            hip = SceneEntityCfg("robot")
            hip.joint_ids = [2, 3]
            self.cfg.params.update(knee_joint_cfg=knee, hip_roll_joint_cfg=hip)
        if bootstrap:
            self.cfg.params["bootstrap_enabled"] = True
        self.state = AlternatingFootstepReward(self.cfg, self.env)
        self.env.reward_manager = SimpleNamespace(get_term_cfg=lambda name: SimpleNamespace(func=self.state))
        self.total = torch.zeros(count, 3)
        self.shape_total = torch.zeros(count, 3)
        self.bootstrap_total = torch.zeros(count)
        self.frame([0, 0], frames=6)

    def frame(self, height, frames=1, contact=None, forces=None):
        for _ in range(frames):
            self.height[:] = torch.as_tensor(height)
            self.position[:, :, 2] = self.height + 0.0408
            self.contacts[:] = (self.height <= 0.005).float() if contact is None else torch.as_tensor(contact)
            self.forces[:, 0, :, 2] = self.contacts * 50 if forces is None else torch.as_tensor(forces)
            self.env.common_step_counter += 1
            gait = self.state(self.env, **self.cfg.params)
            lift = alternating_sole_lift_credit(self.env)
            air = alternating_air_time_credit(self.env)
            self.total += torch.stack((gait, lift, air), dim=1) * self.env.step_dt
            if self.knee_shaping:
                knee = alternating_knee_lift_credit(self.env)
                roll = swing_hip_roll_excess(self.env)
                landing = landing_knee_excess(self.env)
                self.shape_total += torch.stack((knee, roll, landing), dim=1) * self.env.step_dt
            if self.bootstrap:
                self.bootstrap_total += bootstrap_knee_lift_credit(self.env) * self.env.step_dt

    def cycle(self, foot):
        height = [0.0, 0.0]; height[foot] = 0.06
        self.frame(height, frames=10)
        self.frame([0, 0], frames=4)


def check_steps():
    h = StepHarness()
    assert h.total.sum() == 0
    h.cycle(0)
    torch.testing.assert_close(h.state.rewarded_steps, torch.tensor([[1.0, 0.0]] * 2))
    first = h.total.clone()
    h.cycle(0); h.cycle(0)
    torch.testing.assert_close(h.total, first)
    assert (h.state.valid_steps[:, 0] == 3).all() and (h.state.required_foot == 1).all()
    h.cycle(1); h.cycle(0)
    torch.testing.assert_close(h.state.rewarded_steps, torch.tensor([[2.0, 1.0]] * 2))
    assert (h.state.alternations == 2).all() and (h.state.repeated_steps == 2).all()
    print("PASS: repeated same-leg steps earn no gait/lift/air-time credit; L-R-L earns credit", flush=True)

    h = StepHarness(); h.cycle(1)
    assert (h.state.required_foot == 0).all()
    before = h.total.clone(); h.cycle(1)
    torch.testing.assert_close(h.total, before)
    print("PASS: either leg can start, with symmetric ordering rules", flush=True)

    h = StepHarness()
    h.frame([0.06, 0], frames=10)
    # Contact flicker while the foot is physically high is not a landing.
    h.frame([0.06, 0], frames=2, contact=[1, 1])
    h.frame([0.06, 0], frames=30)
    assert h.state.valid_steps.sum() == 0
    torch.testing.assert_close(h.total[:, 1], torch.ones(2))
    h.frame([0, 0], frames=4)  # timeout: not a valid step
    assert h.state.rewarded_steps.sum() == 0
    old_lift = h.total[:, 1].clone(); h.cycle(0)
    torch.testing.assert_close(h.total[:, 1], old_lift)
    assert (h.state.rewarded_steps[:, 0] == 1).all()
    print("PASS: held legs, timeouts, contact flicker and failed retries cannot renew lift credit", flush=True)

    h = StepHarness(); h.frame([0.06, 0.06], frames=12); h.frame([0, 0], frames=4)
    assert h.total.sum() == 0 and h.state.valid_steps.sum() == 0
    h = StepHarness(); h.frame([0.003, 0], frames=15); h.frame([0, 0], frames=4)
    assert h.total.sum() == 0
    h = StepHarness(); h.commands.zero_(); h.cycle(0)
    assert h.total.sum() == 0
    print("PASS: jumping, millimeter foot motion and zero commands do not earn step credit", flush=True)

    h = StepHarness(); h.cycle(0)
    h.state.reset(torch.tensor([0]))
    assert h.state.required_foot.tolist() == [-1, 1]
    assert h.env.extras["log"]["Metrics/gait/left_valid_steps"] == 1
    before = h.total.clone(); h.frame([0, 0], frames=6); h.cycle(0)
    assert h.total[0, 0] > before[0, 0] and h.total[1, 0] == before[1, 0]
    steps = h.state.valid_steps.clone()
    alternating_sole_lift_credit(h.env); alternating_air_time_credit(h.env)
    torch.testing.assert_close(h.state.valid_steps, steps)
    h.state.reset([])
    print("PASS: partial reset, TensorBoard episode metrics and once-per-step shared state", flush=True)


def check_continuous_walking():
    def tracking(h):
        return walking_rewards.track_lin_vel_xy_progress(
            h.env, std=0.2, command_name="base_velocity", gait_reward_name="gait", gait_exploration_fraction=0.25
        )

    h = StepHarness()
    torch.testing.assert_close(tracking(h), torch.full((2,), 0.25))
    for _ in range(4):
        h.cycle(0)
    assert h.state.recent_quality.sum() == 0  # one foot alone cannot qualify
    torch.testing.assert_close(tracking(h), torch.full((2,), 0.25))
    h = StepHarness(); h.cycle(0); h.cycle(1)
    assert ((h.state.recent_quality > 0) & (h.state.recent_quality < 1)).all()
    h.cycle(0); h.cycle(1)
    torch.testing.assert_close(h.state.recent_quality, torch.ones(2))
    torch.testing.assert_close(tracking(h), torch.ones(2))
    totals = h.state.walking_totals.clone()
    h.frame([0, 0], frames=210)  # old pair expires even while motion is commanded
    assert h.state.recent_quality.sum() == 0
    torch.testing.assert_close(tracking(h), torch.full((2,), 0.25))
    torch.testing.assert_close(h.state.walking_totals, totals)
    h.cycle(0); h.cycle(1)
    h.commands.zero_()
    h.env.scene["robot"].data.root_lin_vel_w.zero_()
    h.frame([0, 0])
    torch.testing.assert_close(tracking(h), torch.ones(2))  # standing objective stays intact
    assert h.state.recent_quality.sum() == 0 and h.state._quality_history.sum() == 0
    walking_time = h.state.walking_time.clone()
    h.frame([0, 0], frames=20)
    torch.testing.assert_close(h.state.walking_time, walking_time)
    h.commands[:, 0] = 0.25
    assert h.state.recent_quality.sum() == 0
    h.cycle(0); h.cycle(1); h.cycle(0); h.cycle(1)
    second = h.state._quality_history[:, 1].clone()
    h.state.reset(torch.tensor([0]))
    assert h.state._quality_history[:, 0].sum() == 0
    torch.testing.assert_close(h.state._quality_history[:, 1], second)
    assert h.state.walking_time[0] == 0 and h.state.walking_time[1] > 0
    h = StepHarness(); walking_time = h.state.walking_time.clone()
    h.commands[:] = torch.tensor([0., 0., 0.2])
    h.cycle(0); h.cycle(1)
    torch.testing.assert_close(h.state.walking_time, walking_time)
    assert h.state.walking_totals.sum() == 0
    print("PASS: rolling gait quality rejects unilateral/expired steps, standing and yaw do not inflate walking stats; partial resets isolated", flush=True)


def check_noise_and_export():
    task = "Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift"
    agent = load_cfg_from_registry(task, "rsl_rl_cfg_entry_point")
    obs = TensorDict({"policy": torch.zeros(4, 675), "critic": torch.zeros(4, 720)}, batch_size=[4])
    env = SimpleNamespace(num_envs=4, num_actions=12, device="cpu", get_observations=lambda: obs)
    runner = make_on_policy_runner(env, agent.to_dict(), device="cpu")
    policy = runner.alg.policy
    with torch.no_grad():
        policy.log_std.fill_(8.0)
    policy.log_std.grad = -torch.ones_like(policy.log_std)
    runner.alg.optimizer.step()
    assert (policy.log_std.exp() <= agent.max_noise_std + 1e-6).all()
    with torch.no_grad():
        policy.log_std.fill_(-20.0)
    policy.log_std.grad = torch.ones_like(policy.log_std)
    runner.alg.optimizer.step()
    assert (policy.log_std.exp() >= agent.min_noise_std - 1e-6).all()
    saved = policy.state_dict(); saved["log_std"] = torch.full_like(saved["log_std"], 8.0)
    policy.load_state_dict(saved)
    policy.act(obs)
    assert (policy.action_std <= agent.max_noise_std + 1e-6).all()
    assert torch.isfinite(policy.get_actions_log_prob(policy.action_mean)).all()
    export_dir = "/tmp/p1_alternation_export_check"
    export_policy_as_jit(policy, normalizer=None, path=export_dir, filename="policy.pt")
    exported = torch.jit.load(str(Path(export_dir) / "policy.pt"))
    inputs = torch.randn(2, 675)
    torch.testing.assert_close(exported(inputs), policy.actor(inputs))
    limits = SimpleNamespace(lin_vel_x=(-0.45, 0.55), lin_vel_y=(-0.30, 0.30))
    ranges = SimpleNamespace(lin_vel_x=(0.08, 0.24), lin_vel_y=(-0.06, 0.06), ang_vel_z=(0., 0.))
    command = SimpleNamespace(cfg=SimpleNamespace(ranges=ranges, limit_ranges=limits))
    env.unwrapped = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda name: command),
                                    _balanced_omni_curriculum_state={"stage": 1})
    runner.logger_type = "tensorboard"
    checkpoint = str(Path(export_dir) / "checkpoint_with_stage.pt")
    runner.current_learning_iteration = 7800
    runner.save(checkpoint, {"test_marker": "preserved"})
    info = runner.load(checkpoint, load_optimizer=False)
    assert info["p1_curriculum_stage"] == 1 and info["test_marker"] == "preserved"
    assert info["p1_command_ranges"]["lin_vel_x"] == (0.08, 0.24)
    fresh = make_on_policy_runner(env, agent.to_dict(), device="cpu")
    initialization = initialize_policy_for_finetuning(fresh, checkpoint, noise_std=0.25)
    assert fresh.current_learning_iteration == 0 and not fresh.alg.optimizer.state
    assert initialization["iteration"] == 7800 and initialization["reset_optimizer"]
    torch.testing.assert_close(fresh.alg.policy.actor(inputs), policy.actor(inputs))
    torch.testing.assert_close(fresh.alg.policy.log_std.exp(), torch.full_like(policy.log_std, 0.25))
    print("PASS: std bounds after optimizer steps/checkpoint load; unchanged 675-input JIT export", flush=True)
    print("PASS: checkpoints preserve per-stage command ranges for playback", flush=True)
    print("PASS: fine-tuning preserves policy weights, resets iteration/optimizer and initializes bounded noise", flush=True)


def check_bootstrap_guidance():
    h = StepHarness(knee_shaping=True, bootstrap=True)
    h.joint_pos[:, 0] = 0.18
    h.frame([0, 0], frames=10)
    assert h.bootstrap_total.sum() == 0  # ordinary loaded knee bend gives nothing
    h.joint_pos[:, 1] = 0.18
    h.frame([0, 0], frames=5, forces=[5, 50])
    assert h.bootstrap_total.sum() == 0  # bilateral crouch cannot substitute
    h.joint_pos[:, 1] = 0.0
    h.frame([0.003, 0], frames=5, forces=[5, 50])
    assert (h.bootstrap_total > 0).all()  # signal before the 1 cm main takeoff
    before = h.bootstrap_total.clone()
    h.frame([0.003, 0], frames=10, forces=[5, 50])
    torch.testing.assert_close(h.bootstrap_total, before)

    def mini(h, foot):
        h.joint_pos[:, :2] = 0
        h.joint_pos[:, foot] = 0.10
        height = [0., 0.]; height[foot] = 0.007
        force = [50., 50.]; force[foot] = 0.
        h.frame(height, frames=5, forces=force)
        h.joint_pos[:, :2] = 0
        h.frame([0, 0], frames=4)

    h = StepHarness(knee_shaping=True, bootstrap=True)
    mini(h, 0)
    assert h.state.bootstrap_steps[:, 0].sum() == 2
    assert h.state.valid_steps.sum() == 0  # mini-landings never pass real gait gates
    paid = h.bootstrap_total.clone()
    mini(h, 0)
    torch.testing.assert_close(h.bootstrap_total, paid)
    mini(h, 1); mini(h, 0)
    assert (h.bootstrap_total > paid).all()
    assert (h.state.bootstrap_steps[:, 0] == 2).all() and (h.state.bootstrap_steps[:, 1] == 1).all()

    h = StepHarness(knee_shaping=True, bootstrap=True)
    h.joint_pos[:, 0] = 0.18
    h.frame([0.007, 0], frames=50, forces=[0, 50])
    paid = h.bootstrap_total.clone()
    assert paid.max() <= 1 and h.state.bootstrap_steps.sum() == 0
    h.commands.zero_(); h.frame([0, 0], frames=5)
    h.commands[:, 0] = 0.25
    h.frame([0.007, 0], frames=10, forces=[0, 50])
    torch.testing.assert_close(h.bootstrap_total, paid)
    other = h.state.bootstrap_credit[1].clone()
    h.state.reset(torch.tensor([0]))
    assert h.state.bootstrap_credit[0].sum() == 0
    torch.testing.assert_close(h.state.bootstrap_credit[1], other)
    h = StepHarness(knee_shaping=True, bootstrap=True)
    h.commands.zero_(); h.joint_pos[:, 0] = 0.18
    h.frame([0.007, 0], frames=8, forces=[0, 50])
    assert h.bootstrap_total.sum() == 0
    print("PASS: sub-cm knee/lift bootstrap, unloaded support, alternating finite budgets, timeout/stop protection and real-gait isolation", flush=True)


def check_knee_shaping():
    h = StepHarness(knee_shaping=True)
    h.joint_pos[:, 0] = 0.42
    h.frame([0, 0], frames=10)
    assert h.shape_total.sum() == 0  # bending while planted earns nothing
    h.joint_pos[:, 0] = 0.0
    h.cycle(0)
    assert h.shape_total[:, 0].sum() == 0  # straight-legged lift earns no knee bonus

    h = StepHarness(knee_shaping=True)
    h.joint_pos[:, 0] = 0.42
    h.joint_pos[:, 2] = 0.04
    h.frame([0.06, 0], frames=10)
    h.joint_pos[:, 0] = 0.10  # modest knee bend is allowed at landing
    h.frame([0, 0], frames=4)
    torch.testing.assert_close(h.shape_total[:, 0], torch.ones(2))
    assert h.shape_total[:, 1:].sum() == 0
    assert (h.state.knee_peak_sum[:, 0] == 0.42).all()
    assert (h.state.knee_flexed_steps[:, 0] == 1).all()
    before = h.shape_total[:, 0].clone()
    h.joint_pos[:, 0] = 0.42
    h.cycle(0)
    torch.testing.assert_close(h.shape_total[:, 0], before)
    h.joint_pos[:, 1] = 0.42
    h.joint_pos[:, 0] = 0.10
    h.cycle(1)
    h.joint_pos[:, 0] = 0.42
    h.joint_pos[:, 1] = 0.10
    h.cycle(0)
    torch.testing.assert_close(h.shape_total[:, 0], torch.full((2,), 3.0))
    print("PASS: knee bonus requires real lift and alternating legs; peak angles persist through landing", flush=True)

    crouched = StepHarness(knee_shaping=True)
    crouched.joint_pos[:, :2] = 0.42
    crouched.cycle(0)
    assert crouched.shape_total[:, 0].sum() == 0
    print("PASS: a permanent bilateral knee crouch does not earn swing-knee credit", flush=True)

    straight = StepHarness(knee_shaping=True)
    straight.joint_pos[:, 0] = 0.42
    straight.joint_pos[:, 2] = 0.16
    straight.cycle(0)
    assert (straight.shape_total[:, 0] < 1.0).all()
    assert (straight.shape_total[:, 1:] > 0).all()
    lateral = StepHarness(knee_shaping=True)
    lateral.commands[:, 1] = 0.3
    lateral.joint_pos[:, 0] = 0.42
    lateral.joint_pos[:, 2] = 0.16
    lateral.cycle(0)
    torch.testing.assert_close(lateral.shape_total[:, 0], torch.ones(2))
    assert lateral.shape_total[:, 1].sum() == 0
    print("PASS: excessive swing roll reduces bonus and incurs cost; lateral commands relax allowance", flush=True)

    h = StepHarness(knee_shaping=True)
    h.joint_pos[:, 0] = 0.42
    h.frame([0.06, 0], frames=42)
    before = h.shape_total[:, 0].clone()
    h.frame([0.06, 0], frames=20, contact=[1, 1])
    torch.testing.assert_close(h.shape_total[:, 0], before)
    assert h.shape_total[:, 0].max() <= 1.0 + 1e-6
    h.frame([0, 0], frames=4)
    h.cycle(0)
    torch.testing.assert_close(h.shape_total[:, 0], before)
    h.state.reset(torch.tensor([0]))
    assert h.state.knee_credit[0].sum() == 0 and h.state.knee_credit[1, 0] > 0
    assert "Metrics/gait/left_peak_knee_delta_rad" in h.env.extras["log"]
    print("PASS: holds, contact flicker and retries cannot renew knee bonus; partial resets stay isolated", flush=True)


try:
    check_keyboard_limits()
    check_standing_costs()
    check_steps()
    check_continuous_walking()
    check_bootstrap_guidance()
    check_knee_shaping()
    check_noise_and_export()
    print("PASS: all alternation and continuous-walking checks completed", flush=True)
except BaseException:
    import traceback
    traceback.print_exc()
    raise
app.close()
