"""Check that the walking task removes the standing reward shortcut."""

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks
from unitree_rl_lab.tasks.locomotion import mdp
from unitree_rl_lab.tasks.locomotion.robots.p_pace.agents.rsl_rl_walk_ppo_cfg import PWalkPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.p_pace.velocity_env_cfg import WalkEnvCfg, WalkPlayEnvCfg

cfg = WalkEnvCfg()
agent = PWalkPPORunnerCfg()
assert cfg.commands.base_velocity.ranges.lin_vel_x == (0.18, 0.30)
assert cfg.commands.base_velocity.rel_standing_envs == 0.05
assert agent.policy.init_noise_std == 0.35
assert agent.policy.noise_std_type == "log"
assert agent.algorithm.entropy_coef == 0.0
assert cfg.rewards.track_lin_vel_xy.weight == 2.0
assert cfg.rewards.track_lin_vel_xy.params["std"] == 0.20
assert cfg.curriculum.velocity_levels.params["promote_standing_score"] == 0.0
assert cfg.curriculum.velocity_levels.params["promote_survival_rate"] == 0.85
play = WalkPlayEnvCfg()
assert play.commands.base_velocity.ranges.lin_vel_x == (0.25, 0.25)
assert play.commands.base_velocity.rel_standing_envs == 0.0

term = WalkEnvCfg().rewards.feet_clearance
asset_cfg = term.params["asset_cfg"]
sensor_cfg = term.params["sensor_cfg"]
asset_cfg.body_ids = [0, 1]
sensor_cfg.body_ids = [0, 1]
foot_velocity = torch.zeros(4, 2, 3)
foot_velocity[1:, 0, 0] = 1.0
foot_position = torch.zeros(4, 2, 3)
foot_position[:, :, 2] = 0.1
contact_time = torch.zeros(4, 2)
contact_time[2, 0] = 0.1
commands = torch.full((4, 3), 0.25)
commands[:, 1:] = 0.0
commands[3] = 0.0
robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=foot_position, body_lin_vel_w=foot_velocity))
contact = SimpleNamespace(data=SimpleNamespace(current_contact_time=contact_time))
class Scene:
    sensors = {"contact_forces": contact}
    def __getitem__(self, name):
        return {"robot": robot, "contact_forces": contact}[name]

env = SimpleNamespace(
    scene=Scene(), num_envs=4, device="cpu", step_dt=0.02,
    episode_length_buf=torch.full((4,), 24),
    command_manager=SimpleNamespace(get_command=lambda name: commands),
)
old = mdp.foot_clearance_reward(env, asset_cfg, target_height=0.1, std=0.05, tanh_mult=2.0)
new = term.func(env, **term.params)
assert old[0].item() == 1.0
assert new[0].item() == 0.0
assert new[1].item() > 0.0
assert new[2].item() == 0.0
assert new[3].item() == 0.0
print("PASS: stationary, contact and zero-command feet earn no clearance reward", flush=True)

from unitree_rl_lab.tasks.locomotion.mdp.p_robot import balanced_omni_velocity_levels

count = 512
curriculum_cfg = WalkEnvCfg()
command_term = SimpleNamespace(
    cfg=SimpleNamespace(
        ranges=curriculum_cfg.commands.base_velocity.ranges,
        limit_ranges=curriculum_cfg.commands.base_velocity.limit_ranges,
    ),
    axis_tracking_error_sum=torch.tensor([[0.075, 0.085, 0.55]]).repeat(count, 1) * 40.0,
    axis_tracking_sample_count=torch.full((count, 3), 40.0),
    standing_motion_error_sum=torch.ones(count),
    standing_sample_count=torch.ones(count),
)
curriculum_env = SimpleNamespace(
    num_envs=count, device="cpu", common_step_counter=0,
    episode_length_buf=torch.full((count,), 860), max_episode_length=1000,
    command_manager=SimpleNamespace(get_term=lambda name: command_term),
)
params = curriculum_cfg.curriculum.velocity_levels.params
balanced_omni_velocity_levels(curriculum_env, [], **params)
curriculum_env.common_step_counter = params["update_interval_steps"]
result = balanced_omni_velocity_levels(curriculum_env, list(range(count)), **params)
assert result["direction_x"].item() == 1.0
assert result["direction_y"].item() == 0.0
assert result["direction_yaw"].item() == 0.0
assert command_term.cfg.ranges.lin_vel_x[1] > 0.30
print("PASS: forward curriculum advances despite poor standing/yaw when walking qualifies", flush=True)

cfg.scene.num_envs = 4
env = gym.make("Unitree-P-G1Rewards-PACE-Walk", cfg=cfg)
env.reset()
for _ in range(args.steps):
    _, reward, _, _, _ = env.step(torch.zeros(4, 12, device=env.unwrapped.device))
    assert torch.isfinite(reward).all()
print(f"PASS: walking task 4 environments x {args.steps} steps", flush=True)
env.close()
app.close()
