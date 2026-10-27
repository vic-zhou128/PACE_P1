# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
from importlib.metadata import version
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard-vx", type=float, default=0.15, help="P1 W/S speed magnitude in m/s (default: 0.15).")
parser.add_argument("--keyboard-vy", type=float, default=0.05, help="P1 A/D speed magnitude in m/s (default: 0.05).")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if any(not math.isfinite(value) or value < 0 for value in (args_cli.keyboard_vx, args_cli.keyboard_vy)):
    parser.error("Keyboard speed magnitudes must be finite and nonnegative")
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.envs import DirectMARLEnv, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_runner import make_on_policy_runner
from unitree_rl_lab.utils.p_playback import keyboard_speed_ranges


class KeyboardVelocityController:
    """Hold WASD for base-frame translation; the command term handles acceleration."""

    def __init__(self, command_term, limits):
        import carb.input
        import omni.appwindow
        import omni.ui as ui

        self.command_term = command_term
        self.limits = limits
        self.pressed = set()
        self.key_inputs = {key: getattr(carb.input.KeyboardInput, key) for key in ("W", "A", "S", "D")}
        self.last_target = None
        self.last_status_time = 0.0
        self.last_log_time = 0.0
        window = omni.appwindow.get_default_app_window()
        if window is None or window.get_keyboard() is None:
            raise RuntimeError("Isaac Sim has no keyboard window; use the graphical play interface")
        self.keyboard = window.get_keyboard()
        self.input = carb.input.acquire_input_interface()
        self.event_types = carb.input.KeyboardEventType
        self.subscription = self.input.subscribe_to_keyboard_events(self.keyboard, self._on_key)
        self.command_term.enable_manual_control()
        self.status_window = ui.Window(
            "P1 Keyboard Control", width=430, height=210, flags=ui.WINDOW_FLAGS_NO_FOCUS_ON_APPEARING
        )
        with self.status_window.frame:
            with ui.VStack(spacing=4):
                ui.Label("Hold W/S: forward/backward; A/D: left/right")
                ui.Label("Click the viewport and use an English keyboard layout.")
                ui.Label(f"Speed: forward {limits.lin_vel_x[1]:.2f}, backward {limits.lin_vel_x[0]:.2f}, lateral {limits.lin_vel_y[1]:.2f} m/s")
                self.keys_label = ui.Label("Keys: none")
                self.target_label = ui.Label("Target: vx=+0.00, vy=+0.00 m/s")
                self.command_label = ui.Label("Command: vx=+0.00, vy=+0.00 m/s")
                self.measured_label = ui.Label("Measured: vx=+0.00, vy=+0.00 m/s")

    def _on_key(self, event, *args, **kwargs):
        key = event.input.name
        if key in self.key_inputs:
            if event.type == self.event_types.KEY_PRESS:
                self.pressed.add(key)
                print(f"[P1 keyboard] {key} pressed", flush=True)
            elif event.type == self.event_types.KEY_RELEASE:
                self.pressed.discard(key)
                print(f"[P1 keyboard] {key} released", flush=True)
        return True

    def update(self):
        # Poll the held state as well as receiving events. This avoids relying
        # on a single key-press callback and clears keys when they are released.
        self.pressed = {
            key for key, value in self.key_inputs.items() if self.input.get_keyboard_value(self.keyboard, value) > 0.5
        }
        forward = int("W" in self.pressed) - int("S" in self.pressed)
        lateral = int("A" in self.pressed) - int("D" in self.pressed)
        x = self.limits.lin_vel_x[1] if forward > 0 else self.limits.lin_vel_x[0] if forward < 0 else 0.0
        y = self.limits.lin_vel_y[1] if lateral > 0 else self.limits.lin_vel_y[0] if lateral < 0 else 0.0
        target = (x, y, 0.0)
        self.command_term.set_manual_target(target)
        if target != self.last_target:
            print(f"[P1 keyboard] Target vx={x:+.2f}, vy={y:+.2f} m/s", flush=True)
            self.last_target = target

        now = time.monotonic()
        if now - self.last_status_time >= 0.1:
            command = self.command_term.command[0].detach().cpu().tolist()
            measured = self.command_term.robot.data.root_lin_vel_b[0, :2].detach().cpu().tolist()
            self.keys_label.text = f"Keys: {'+'.join(sorted(self.pressed)) or 'none'}"
            self.target_label.text = f"Target: vx={x:+.2f}, vy={y:+.2f} m/s"
            self.command_label.text = f"Command: vx={command[0]:+.2f}, vy={command[1]:+.2f} m/s"
            self.measured_label.text = f"Measured: vx={measured[0]:+.2f}, vy={measured[1]:+.2f} m/s"
            self.last_status_time = now
            if now - self.last_log_time >= 1.0 and (self.pressed or abs(command[0]) + abs(command[1]) > 0.01):
                print(
                    f"[P1 keyboard] Keys={'+'.join(sorted(self.pressed)) or 'none'}, "
                    f"command=({command[0]:+.2f}, {command[1]:+.2f}), "
                    f"measured=({measured[0]:+.2f}, {measured[1]:+.2f}) m/s",
                    flush=True,
                )
                self.last_log_time = now

    def close(self):
        self.input.unsubscribe_to_keyboard_events(self.keyboard, self.subscription)
        self.status_window.visible = False


def main():
    """Play with RSL-RL agent."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg: RslRlOnPolicyRunnerCfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)
    keyboard_enabled = not args_cli.headless and args_cli.task.startswith("Unitree-P-")
    if keyboard_enabled:
        # The first reset must produce zero commands, before any keyboard event.
        command_cfg = env_cfg.commands.base_velocity
        command_cfg.ranges.lin_vel_x = (0.0, 0.0)
        command_cfg.ranges.lin_vel_y = (0.0, 0.0)
        command_cfg.ranges.ang_vel_z = (0.0, 0.0)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

        resume_path = get_published_pretrained_checkpoint("rsl_rl", args_cli.task)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    keyboard = None

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if not hasattr(agent_cfg, "class_name") or agent_cfg.class_name == "OnPolicyRunner":
        runner = make_on_policy_runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        from rsl_rl.runners import DistillationRunner

        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    checkpoint_info = runner.load(resume_path)
    if keyboard_enabled:
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        if not hasattr(command_term, "enable_manual_control"):
            raise TypeError("Keyboard control requires the P OmniVelocityCommand term")
        trained_ranges = (checkpoint_info or {}).get("p1_command_ranges")
        if trained_ranges is None:
            config_path = Path(log_dir) / "params" / "env.yaml"
            if config_path.is_file():
                import yaml
                # BaseLoader treats Python tuple tags as plain sequences and
                # never constructs Python objects from the saved YAML.
                with config_path.open() as stream:
                    saved_config = yaml.load(stream, Loader=yaml.BaseLoader)
                trained_ranges = saved_config["commands"]["base_velocity"]["ranges"]
                print("[INFO] No stage metadata in this checkpoint; keyboard uses its initial training ranges.")
        keyboard_limits = keyboard_speed_ranges(
            env_cfg.commands.base_velocity.limit_ranges, args_cli.keyboard_vx, args_cli.keyboard_vy, trained_ranges
        )
        keyboard = KeyboardVelocityController(command_term, keyboard_limits)
        print(f"[INFO] Keyboard speeds: vx={keyboard_limits.lin_vel_x}, vy={keyboard_limits.lin_vel_y} m/s")
        print("[INFO] WASD: hold W/S for forward/backward, A/D for left/right; release to stop.")
        print("[INFO] A direction with a zero speed limit has not been enabled for this checkpoint.")
        print("[INFO] Click the Isaac Sim viewport to give it keyboard focus.")

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    if version("rsl-rl-lib").startswith("2.3."):
        obs, _ = env.get_observations()
    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            # run everything in inference mode
            with torch.inference_mode():
                if keyboard is not None:
                    keyboard.update()
                # agent stepping
                actions = policy(obs)
                # env stepping
                obs, _, _, _ = env.step(actions)
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break

            # time delay for real-time evaluation
            sleep_time = dt - (time.time() - start_time)
            if args_cli.real_time and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if keyboard is not None:
            keyboard.close()
        env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
