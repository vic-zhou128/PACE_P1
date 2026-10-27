"""Measure sole lift and body motion for a P no-phase walking checkpoint."""

import argparse
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "unitree_rl_lab"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--task", default="Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-DenseLift")
parser.add_argument("--steps", type=int, default=1000)
parser.add_argument("--warmup-steps", type=int, default=100)
parser.add_argument("--vx", type=float, default=0.25)
parser.add_argument("--vy", type=float, default=0.0)
parser.add_argument("--yaw", type=float, default=0.0)
parser.add_argument("--seed", type=int, default=42)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.steps <= args.warmup_steps + 1 or args.warmup_steps < 0:
    parser.error("--steps must exceed --warmup-steps by at least two")
app = AppLauncher(args).app

import gymnasium as gym
import torch
from isaaclab.utils.math import quat_apply
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg
from unitree_rl_lab.utils.rsl_rl_runner import make_on_policy_runner


def percentile(values, fraction):
    values = sorted(values)
    return values[round((len(values) - 1) * fraction)] if values else float("nan")


def completed_lift_cycles(heights, contacts, episode_ids, step_dt):
    """Count real takeoff/landing cycles, tolerating brief contact-signal flicker."""
    events = []
    cycle_peaks = []
    peak_indices = []
    previous_episode = None
    active = armed = contact_seen = False
    low_steps = high_steps = longest_high = 0
    peak = 0.0
    peak_index = 0
    start = 0
    for index, (height, contact, episode) in enumerate(zip(heights, contacts, episode_ids)):
        if episode != previous_episode:
            active = armed = contact_seen = False
            low_steps = high_steps = 0
        previous_episode = episode
        high_steps = high_steps + 1 if height >= 0.02 else 0
        longest_high = max(longest_high, high_steps)
        if height <= 0.005:
            low_steps += 1
            contact_seen = contact_seen or contact
        else:
            low_steps = 0
            contact_seen = False
        grounded = low_steps * step_dt >= 0.06 - 1.0e-6 and contact_seen
        if active:
            if height > peak:
                peak = height
                peak_index = index
            if grounded:
                duration = (index - low_steps + 1 - start) * step_dt
                if peak >= 0.02 and duration >= 0.08:
                    events.append((episode, index))
                    cycle_peaks.append(peak)
                    peak_indices.append(peak_index)
                active = False
                armed = True
        elif grounded:
            armed = True
        elif armed and height >= 0.01:
            active = True
            armed = False
            peak = height
            peak_index = index
            start = index
    return events, longest_high * step_dt, cycle_peaks, peak_indices


cfg = parse_env_cfg(args.task, num_envs=1, entry_point_key="play_env_cfg_entry_point")
cfg.scene.num_envs = 1
cfg.seed = args.seed
cfg.commands.base_velocity.debug_vis = False
cfg.episode_length_s = max(cfg.episode_length_s, (args.steps + 1) * cfg.decimation * cfg.sim.dt)
cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
env = gym.make(args.task, cfg=cfg)
env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
runner = make_on_policy_runner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
runner.load(str(args.checkpoint.resolve()))
policy = runner.get_inference_policy(device=env.unwrapped.device)
robot = env.unwrapped.scene["robot"]
sensor = env.unwrapped.scene.sensors["contact_forces"]
body_ids, _ = robot.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
contact_ids, _ = sensor.find_bodies(["left_ankle_roll_link", "right_ankle_roll_link"], preserve_order=True)
knee_ids, _ = robot.find_joints(["left_knee_pitch_joint", "right_knee_pitch_link"], preserve_order=True)
hip_roll_ids, _ = robot.find_joints(["left_hip_roll_link", "right_hip_roll_link"], preserve_order=True)
command_term = env.unwrapped.command_manager.get_term("base_velocity")
command_term.enable_manual_control()
command_term.set_manual_target((0.0, 0.0, 0.0))
observations = env.get_observations()

centers = torch.tensor(
    [(-0.105, 0.045, -0.0328), (-0.105, -0.045, -0.0328),
     (0.065, 0.045, -0.0328), (0.065, -0.045, -0.0328)],
    device=env.unwrapped.device,
)
air_peaks = [[], []]
active_peak = [None, None]
active_steps = [0, 0]
air_heights = [[], []]
all_heights = [[], []]
all_contacts = [[], []]
all_knee_angles = []
all_knee_offsets = []
all_hip_roll_offsets = []
all_body_roll = []
sample_episodes = []
air_samples = [0, 0]
tilt = []
ang_xy = []
base_heights = []
action_deltas = []
measured_velocities = []
applied_commands = []
segment_displacement = torch.zeros(2, device=env.unwrapped.device)
last_root_xy = None
last_action = None
reset_count = 0
step_dt = env.unwrapped.step_dt

with torch.inference_mode():
    for step in range(args.steps):
        if step == args.warmup_steps:
            command_term.set_manual_target((args.vx, args.vy, args.yaw))
        actions = policy(observations)
        observations, _, dones, _ = env.step(actions)
        done = bool(dones[0].item())
        reset_count += int(done)
        if step < args.warmup_steps or done:
            active_peak = [None, None]
            active_steps = [0, 0]
            last_action = None
            last_root_xy = None
            continue

        measured_velocities.append(robot.data.root_lin_vel_b[0, :2].cpu().tolist())
        sample_episodes.append(reset_count)
        joint_pos = robot.data.joint_pos[0]
        joint_offset = joint_pos - robot.data.default_joint_pos[0]
        all_knee_angles.append(joint_pos[knee_ids].cpu().tolist())
        all_knee_offsets.append(joint_offset[knee_ids].cpu().tolist())
        all_hip_roll_offsets.append(joint_offset[hip_roll_ids].cpu().tolist())
        applied_commands.append(command_term.command[0, :2].cpu().tolist())
        root_xy = robot.data.root_pos_w[0, :2]
        if last_root_xy is not None:
            segment_displacement += root_xy - last_root_xy
        last_root_xy = root_xy.clone()

        pos = robot.data.body_pos_w[:, body_ids]
        quat = robot.data.body_quat_w[:, body_ids]
        rotated = quat_apply(
            quat.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4),
            centers.view(1, 1, 4, 3).expand(1, 2, -1, -1).reshape(-1, 3),
        ).reshape(1, 2, 4, 3)
        clearance = pos[..., 2] + rotated[..., 2].amin(dim=-1) - 0.008
        clearance -= env.unwrapped.scene.env_origins[:, 2].unsqueeze(1)
        contact = sensor.data.current_contact_time[:, contact_ids] > 0.0
        for foot in range(2):
            height = float(clearance[0, foot].item())
            all_heights[foot].append(height)
            all_contacts[foot].append(bool(contact[0, foot].item()))
            if not bool(contact[0, foot].item()):
                air_samples[foot] += 1
                active_peak[foot] = height if active_peak[foot] is None else max(active_peak[foot], height)
                active_steps[foot] += 1
                air_heights[foot].append(height)
            else:
                if active_peak[foot] is not None and active_steps[foot] * step_dt >= 0.08:
                    air_peaks[foot].append(active_peak[foot])
                active_peak[foot] = None
                active_steps[foot] = 0

        gravity = robot.data.projected_gravity_b[0]
        all_body_roll.append(float(torch.atan2(-gravity[1], -gravity[2]).item()))
        tilt.append(float(torch.linalg.norm(gravity[:2]).item()))
        ang_xy.append(float(torch.linalg.norm(robot.data.root_ang_vel_b[0, :2]).item()))
        base_heights.append(float(robot.data.root_pos_w[0, 2].item()))
        if last_action is not None:
            action_deltas.append(float(torch.linalg.norm(actions[0] - last_action).item()))
        last_action = actions[0].clone()

print(f"checkpoint: {args.checkpoint.resolve()}")
print(f"command: vx={args.vx:.3f}, vy={args.vy:.3f}, yaw={args.yaw:.3f}; steps={args.steps}; warmup={args.warmup_steps}; resets={reset_count}")
for axis, name in enumerate(("x", "y")):
    measured = [v[axis] for v in measured_velocities]
    commanded = [v[axis] for v in applied_commands]
    error = statistics.mean(abs(a - b) for a, b in zip(measured, commanded))
    print(f"{name} velocity: command_mean={statistics.mean(commanded):.3f} m/s, measured_mean={statistics.mean(measured):.3f} m/s, mean_abs_error={error:.3f} m/s")
print(f"world displacement excluding resets: {segment_displacement.cpu().tolist()} m")
completed_steps = []
for foot, name in enumerate(("left", "right")):
    peaks = air_peaks[foot]
    heights = air_heights[foot]
    genuine = [peak for peak in peaks if peak >= 0.02]
    print(f"{name} contact-free episodes >=80 ms: n={len(peaks)}, >=2 cm={len(genuine)}, >=6 cm={sum(peak >= 0.06 for peak in peaks)}")
    print(f"{name} episode peaks: median={statistics.median(peaks) if peaks else float('nan'):.3f} m, p10={percentile(peaks, 0.1):.3f} m, p90={percentile(peaks, 0.9):.3f} m")
    print(f"{name} airborne height: p50={percentile(heights, 0.5):.3f} m, p90={percentile(heights, 0.9):.3f} m")
    print(f"{name} all-frame height: p99={percentile(all_heights[foot], 0.99):.3f} m, max={max(all_heights[foot]):.3f} m; contact-free fraction={air_samples[foot] / len(all_heights[foot]):.3f}")
    events, longest_high, cycle_peaks, peak_indices = completed_lift_cycles(
        all_heights[foot], all_contacts[foot], sample_episodes, step_dt
    )
    completed_steps.extend((episode, index, foot) for episode, index in events)
    print(f"{name} physical lift cycles >=2 cm: {len(events)}; longest continuously >=2 cm: {longest_high:.2f} s")
    print(
        f"{name} physical cycle peaks: "
        f"median={percentile(cycle_peaks, 0.5):.4f} m, "
        f"p10={percentile(cycle_peaks, 0.1):.4f} m, "
        f"p90={percentile(cycle_peaks, 0.9):.4f} m; "
        f">=6 cm={sum(peak >= 0.06 for peak in cycle_peaks)}/{len(cycle_peaks)}"
    )
    print(
        f"{name} at physical height peaks (median): "
        f"knee={percentile([all_knee_angles[i][foot] for i in peak_indices], 0.5):.4f} rad, "
        f"knee_delta_default={percentile([all_knee_offsets[i][foot] for i in peak_indices], 0.5):.4f} rad, "
        f"knee_delta_swing_minus_support="
        f"{percentile([all_knee_offsets[i][foot] - all_knee_offsets[i][1-foot] for i in peak_indices], 0.5):.4f} rad, "
        f"hip_roll_delta_default={percentile([all_hip_roll_offsets[i][foot] for i in peak_indices], 0.5):.4f} rad, "
        f"body_roll_abs={percentile([abs(all_body_roll[i]) for i in peak_indices], 0.5):.4f} rad"
    )
completed_steps.sort()
pairs = [(a, b) for a, b in zip(completed_steps, completed_steps[1:]) if a[0] == b[0]]
alternations = sum(a[2] != b[2] for a, b in pairs)
sequence = ''.join('L' if foot == 0 else 'R' for _, _, foot in completed_steps)
ratio = alternations / len(pairs) if pairs else float('nan')
print(f"physical lift sequence: {sequence or 'none'}; alternating transitions={alternations}/{len(pairs)}, ratio={ratio:.3f}")
print(f"body tilt sin(angle): rms={(statistics.mean(v*v for v in tilt))**0.5:.4f}, p90={percentile(tilt, 0.9):.4f}")
print(f"body roll/pitch angular speed: rms={(statistics.mean(v*v for v in ang_xy))**0.5:.3f} rad/s, p90={percentile(ang_xy, 0.9):.3f} rad/s")
print(f"base height: mean={statistics.mean(base_heights):.3f} m, std={statistics.pstdev(base_heights):.3f} m")
print(f"action vector step change: rms={(statistics.mean(v*v for v in action_deltas))**0.5:.3f}, p90={percentile(action_deltas, 0.9):.3f}")
env.close()
app.close()
