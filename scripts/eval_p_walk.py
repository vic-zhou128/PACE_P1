"""Measure fixed-command forward walking from a P checkpoint."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--num_envs", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import torch
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

import unitree_rl_lab.tasks
from unitree_rl_lab.tasks.locomotion.robots.p_pace.agents.rsl_rl_walk_ppo_cfg import PWalkPPORunnerCfg
from unitree_rl_lab.tasks.locomotion.robots.p_pace.velocity_env_cfg import WalkPlayEnvCfg

agent_cfg = PWalkPPORunnerCfg()
env_cfg = WalkPlayEnvCfg()
env_cfg.scene.num_envs = args.num_envs
env = gym.make("Unitree-P-G1Rewards-PACE-Walk", cfg=env_cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(str(args.checkpoint.resolve()))
policy = runner.get_inference_policy(device=env.unwrapped.device)
observations = env.get_observations()
robot = env.unwrapped.scene["robot"]
command_manager = env.unwrapped.command_manager
speeds = []
commands = []
reset_count = 0
with torch.inference_mode():
    for step in range(args.steps):
        actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        reset_count += int(dones.sum().item())
        if step >= 50:
            active = ~dones.bool().flatten()
            speeds.extend(robot.data.root_lin_vel_b[active, 0].cpu().tolist())
            commands.extend(command_manager.get_command("base_velocity")[active, 0].cpu().tolist())

if not speeds:
    raise RuntimeError("No uninterrupted walking samples were collected")
print(f"checkpoint: {args.checkpoint.resolve()}")
print(f"steps: {args.steps}, environments: {args.num_envs}, resets: {reset_count}")
print(f"mean forward command: {sum(commands) / len(commands):.3f} m/s")
print(f"mean measured forward speed: {sum(speeds) / len(speeds):.3f} m/s")
print(f"mean absolute forward error: {sum(abs(command - speed) for command, speed in zip(commands, speeds)) / len(speeds):.3f} m/s")
env.close()
app.close()
