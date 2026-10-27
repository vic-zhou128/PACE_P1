"""Exact snapshots of the phased robot-p velocity and push curricula."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _apply_bootstrap_schedule(env, state, tracking_term, bootstrap_term, start_floor, end_floor):
    progress = state["bootstrap_progress"]
    floor = start_floor + (end_floor - start_floor) * progress
    cfg = env.reward_manager.get_term_cfg(tracking_term)
    cfg.params["gait_exploration_fraction"] = floor
    env.reward_manager.set_term_cfg(tracking_term, cfg)
    if bootstrap_term in env.reward_manager.active_terms:
        cfg = env.reward_manager.get_term_cfg(bootstrap_term)
        cfg.weight = state["bootstrap_base_weight"] * (1.0 - progress)
        env.reward_manager.set_term_cfg(bootstrap_term, cfg)
    state["velocity_reward_floor"] = floor


def _apply_locomotion_stage(env, state, stages, reward_terms, push_term_name, physical_gait_term_name=None):
    """Apply a stage to future command samples and the live reward manager."""
    stage = stages[state["stage"]]
    command = env.command_manager.get_term("base_velocity")
    command.cfg.rel_standing_envs = stage["standing_fraction"]
    command.cfg.mode_probabilities = tuple(stage["mode_probabilities"])
    for name, limits in zip(("lin_vel_x", "lin_vel_y", "ang_vel_z"), stage["ranges"]):
        setattr(command.cfg.ranges, name, tuple(limits))
    state["initial_ranges"] = [tuple(x) for x in stage["ranges"]]
    if physical_gait_term_name is not None:
        gait = env.reward_manager.get_term_cfg(physical_gait_term_name).func
        gait.quality_foot_rate_hz = stage["gait_min_foot_rate_hz"]
        gait.quality_height_m = stage["gait_min_height_m"]
    for name in reward_terms:
        if name in state["stage_base_weights"]:
            cfg = env.reward_manager.get_term_cfg(name)
            cfg.weight = state["stage_base_weights"][name] * stage["lift_reward_scale"]
            env.reward_manager.set_term_cfg(name, cfg)
    if push_term_name is not None:
        cfg = env.event_manager.get_term_cfg(push_term_name)
        for axis, speed in zip(("x", "y"), stage["push_velocity"]):
            cfg.params["velocity_range"][axis] = (-speed, speed)
        env.event_manager.set_term_cfg(push_term_name, cfg)


def balanced_omni_velocity_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    update_interval_steps: int = 10000,
    range_steps: tuple[float, float, float] = (0.03, 0.02, 0.04),
    tracking_stds: tuple[float, float, float] = (0.18, 0.14, 0.22),
    promote_tracking_score: float = 0.72,
    demote_tracking_score: float = 0.45,
    promote_standing_score: float = 0.72,
    demote_standing_score: float = 0.45,
    gait_reward_term_name: str | None = None,
    promote_gait_score: float = 0.18,
    demote_gait_score: float = 0.07,
    gait_curriculum_state_attr: str | None = None,
    required_gait_level: int = 0,
    standing_std: float = 0.10,
    promote_survival_rate: float = 0.96,
    demote_survival_rate: float = 0.82,
    min_axis_samples: int = 20000,
    min_standing_samples: int = 10000,
    min_completed_episodes: int = 512,
    stages: tuple[dict, ...] | None = None,
    stage_reward_terms: tuple[str, ...] = ("gait", "feet_clearance", "feet_air_time", "swing_knee_flexion"),
    stage_push_term_name: str | None = None,
    stage_promote_windows: int = 2,
    stage_demote_windows: int = 2,
    min_stopping_samples: int = 50,
    promote_stable_fraction: float = 0.80,
    promote_stopping_success: float = 0.80,
    physical_gait_term_name: str | None = None,
    gait_min_moving_seconds: float = 200.0,
    gait_min_transitions: int = 100,
    gait_min_alternation: float = 0.90,
    gait_min_foot_rate_hz: float = 0.4,
    gait_min_height_m: float = 0.025,
    bootstrap_schedule: bool = False,
    bootstrap_tracking_term: str = "track_lin_vel_xy",
    bootstrap_reward_term: str = "bootstrap_knee_lift",
    bootstrap_velocity_floor_start: float = 1.0,
    bootstrap_velocity_floor_end: float = 0.25,
    bootstrap_adaptation_rate: float = 0.25,
) -> dict[str, torch.Tensor]:
    """Adapt each SE(2) command axis using tracking, standing and gait quality.

    Axis scores include only samples where that axis was commanded. This avoids
    inflating lateral and yaw scores with easy zero-command samples. Standing
    stability, true single support and survival gate every promotion and can
    roll difficulty back.
    """
    command_term = env.command_manager.get_term("base_velocity")
    ranges = command_term.cfg.ranges
    limit_ranges = command_term.cfg.limit_ranges
    range_names = ("lin_vel_x", "lin_vel_y", "ang_vel_z")

    if not hasattr(command_term, "axis_tracking_error_sum"):
        raise AttributeError("balanced_omni_velocity_levels requires OmniVelocityCommand tracking buffers.")
    if any(value <= 0.0 for value in tracking_stds) or standing_std <= 0.0:
        raise ValueError("tracking_stds and standing_std must be positive.")
    if required_gait_level < 0:
        raise ValueError("required_gait_level must be non-negative.")
    if bootstrap_schedule:
        if physical_gait_term_name is None:
            raise ValueError("Bootstrap schedule requires measured physical gait quality")
        if not 0 < bootstrap_velocity_floor_end <= bootstrap_velocity_floor_start <= 1:
            raise ValueError("Invalid bootstrap velocity reward floors")
        if not 0 < bootstrap_adaptation_rate <= 1:
            raise ValueError("Invalid bootstrap adaptation rate")
    if physical_gait_term_name is not None:
        if gait_reward_term_name is not None:
            raise ValueError("Choose physical gait statistics or reward-based gait scoring")
        if min(gait_min_moving_seconds, gait_min_transitions, gait_min_foot_rate_hz, gait_min_height_m) <= 0:
            raise ValueError("Physical gait sample and quality thresholds must be positive")
        if not 0 < gait_min_alternation <= 1:
            raise ValueError("Gait alternation threshold must be in (0, 1]")
    if stages is not None:
        if not stages or min(stage_promote_windows, stage_demote_windows, min_stopping_samples) <= 0:
            raise ValueError("Stages, consecutive-window counts and stopping samples must be positive")
        for stage in stages:
            probs = stage["mode_probabilities"]
            if len(probs) != 4 or min(probs) < 0 or abs(sum(probs) - 1) > 1e-6:
                raise ValueError("Each stage needs four mode probabilities summing to one")
            if not 0 <= stage["standing_fraction"] < 1 or not 0 < stage["lift_reward_scale"] <= 1:
                raise ValueError("Invalid stage standing fraction or lift reward scale")
            if len(stage["ranges"]) != 3 or any(low > high for low, high in stage["ranges"]):
                raise ValueError("Each stage needs ordered x/y/yaw command ranges")
            if physical_gait_term_name is not None and min(stage["gait_min_foot_rate_hz"], stage["gait_min_height_m"]) <= 0:
                raise ValueError("Each stage needs positive physical foot-rate and height thresholds")

    if not hasattr(env, "_balanced_omni_curriculum_state"):
        env._balanced_omni_curriculum_state = {
            "axis_error_sum": [0.0, 0.0, 0.0],
            "axis_samples": [0.0, 0.0, 0.0],
            "standing_error_sum": 0.0,
            "standing_samples": 0.0,
            "gait_score_sum": 0.0,
            "gait_episode_count": 0,
            "survival_sum": 0.0,
            "episode_count": 0,
            "last_update_step": int(env.common_step_counter),
            "tracking_scores": [0.0, 0.0, 0.0],
            "standing_score": 0.0,
            "gait_score": 0.0,
            "survival_rate": 0.0,
            "directions": [0.0, 0.0, 0.0],
            "direction": 0.0,
            "promote_ready": False,
            "demote_ready": False,
            "initial_ranges": [tuple(float(v) for v in getattr(ranges, name)) for name in range_names],
            "stable_samples": 0.0,
            "stopping_attempts": 0.0,
            "stopping_successes": 0.0,
            "stable_fraction": 0.0,
            "stopping_success": 0.0,
            "stage": 0,
            "good_windows": 0,
            "bad_windows": 0,
            "walking_seconds": 0.0,
            "walking_totals": [0.0] * 6,
            "walking_rates": [0.0, 0.0],
            "walking_heights": [0.0, 0.0],
            "walking_alternation": 0.0,
            "evaluated_walking_seconds": 0.0,
            "evaluated_walking_transitions": 0.0,
            "bootstrap_progress": 0.0,
            "velocity_reward_floor": bootstrap_velocity_floor_start,
        }
        if stages is not None:
            state = env._balanced_omni_curriculum_state
            state["stage_base_weights"] = {
                name: float(env.reward_manager.get_term_cfg(name).weight)
                for name in stage_reward_terms if name in env.reward_manager.active_terms
            }
            _apply_locomotion_stage(env, state, stages, stage_reward_terms, stage_push_term_name, physical_gait_term_name)
        if bootstrap_schedule:
            state = env._balanced_omni_curriculum_state
            state["bootstrap_base_weight"] = (
                float(env.reward_manager.get_term_cfg(bootstrap_reward_term).weight)
                if bootstrap_reward_term in env.reward_manager.active_terms else 0.0
            )
            _apply_bootstrap_schedule(env, state, bootstrap_tracking_term, bootstrap_reward_term,
                                      bootstrap_velocity_floor_start, bootstrap_velocity_floor_end)
    state = env._balanced_omni_curriculum_state

    if isinstance(env_ids, slice):
        resolved_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
    else:
        resolved_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    episode_lengths = env.episode_length_buf[resolved_ids]
    valid = episode_lengths > 0
    if torch.any(valid):
        valid_ids = resolved_ids[valid]
        axis_errors = command_term.axis_tracking_error_sum[valid_ids].sum(dim=0)
        axis_samples = command_term.axis_tracking_sample_count[valid_ids].sum(dim=0)
        for axis in range(3):
            state["axis_error_sum"][axis] += float(axis_errors[axis].item())
            state["axis_samples"][axis] += float(axis_samples[axis].item())
        state["standing_error_sum"] += float(command_term.standing_motion_error_sum[valid_ids].sum().item())
        state["standing_samples"] += float(command_term.standing_sample_count[valid_ids].sum().item())
        if stages is not None:
            state["stable_samples"] += float(command_term.standing_stable_samples[valid_ids].sum().item())
            state["stopping_attempts"] += float(command_term.stopping_attempts[valid_ids].sum().item())
            state["stopping_successes"] += float(command_term.stopping_successes[valid_ids].sum().item())
        if physical_gait_term_name is not None:
            gait = env.reward_manager.get_term_cfg(physical_gait_term_name).func
            state["walking_seconds"] += float(gait.walking_time[valid_ids].sum().item())
            totals = gait.walking_totals[valid_ids].sum(dim=0).tolist()
            state["walking_totals"] = [a + b for a, b in zip(state["walking_totals"], totals)]
        if gait_reward_term_name is not None:
            reward_cfg = env.reward_manager.get_term_cfg(gait_reward_term_name)
            reward_weight = float(reward_cfg.weight)
            if reward_weight <= 0.0:
                raise ValueError("gait_reward_term_name must refer to a positive reward term.")
            episode_seconds = episode_lengths[valid].float() * env.step_dt
            gait_scores = env.reward_manager._episode_sums[gait_reward_term_name][valid_ids]
            gait_scores = gait_scores / torch.clamp(episode_seconds * reward_weight, min=1.0e-6)
            state["gait_score_sum"] += float(torch.clamp(gait_scores, 0.0, 1.0).sum().item())
            state["gait_episode_count"] += int(valid_ids.numel())
        survival = episode_lengths[valid].float() / float(env.max_episode_length)
        state["survival_sum"] += float(torch.clamp(survival, 0.0, 1.0).sum().item())
        state["episode_count"] += int(valid_ids.numel())

    active_axes = [True, True, True]
    if stages is not None:
        probs = stages[state["stage"]]["mode_probabilities"]
        active_axes = [probs[axis] > 0 or probs[3] > 0 for axis in range(3)]
        active_axes = [active and low != high for active, (low, high) in zip(active_axes, stages[state["stage"]]["ranges"])]
    enough_samples = all(value >= min_axis_samples for value, active in zip(state["axis_samples"], active_axes) if active)
    enough_samples &= state["standing_samples"] >= min_standing_samples
    enough_samples &= state["episode_count"] >= min_completed_episodes
    if stages is not None:
        enough_samples &= state["stopping_attempts"] >= min_stopping_samples
    if gait_reward_term_name is not None:
        enough_samples &= state["gait_episode_count"] >= min_completed_episodes
    if physical_gait_term_name is not None:
        enough_samples &= state["walking_seconds"] >= gait_min_moving_seconds
    steps_since_update = int(env.common_step_counter) - state["last_update_step"]
    if steps_since_update >= update_interval_steps and enough_samples:
        tracking_scores = []
        for axis in range(3):
            mean_error = state["axis_error_sum"][axis] / max(state["axis_samples"][axis], 1.0)
            tracking_scores.append(math.exp(-mean_error**2 / tracking_stds[axis] ** 2))
        standing_error = state["standing_error_sum"] / max(state["standing_samples"], 1.0)
        standing_score = math.exp(-standing_error**2 / standing_std**2)
        gait_score = (
            state["gait_score_sum"] / max(state["gait_episode_count"], 1)
            if gait_reward_term_name is not None
            else 1.0
        )
        if physical_gait_term_name is not None:
            counts = state["walking_totals"][:2]
            heights = state["walking_totals"][2:4]
            transitions, alternations = state["walking_totals"][4:]
            state["walking_rates"] = [n / max(state["walking_seconds"], 1e-6) for n in counts]
            state["walking_heights"] = [h / max(n, 1.0) for h, n in zip(heights, counts)]
            state["walking_alternation"] = alternations / max(transitions, 1.0)
            state["evaluated_walking_seconds"] = state["walking_seconds"]
            state["evaluated_walking_transitions"] = transitions
            stage = stages[state["stage"]] if stages is not None else {}
            rate_target = stage.get("gait_min_foot_rate_hz", gait_min_foot_rate_hz)
            height_target = stage.get("gait_min_height_m", gait_min_height_m)
            gait_score = min(
                1.0, min(state["walking_rates"]) / rate_target,
                min(state["walking_heights"]) / height_target,
                state["walking_alternation"] / gait_min_alternation,
                transitions / gait_min_transitions,
            )
        if bootstrap_schedule:
            # Only measured completed-step quality can retire the starting
            # signal. More iterations or better standing alone cannot do it.
            target = max(0.0, min(1.0, (gait_score - 0.8) / 0.2))
            state["bootstrap_progress"] += bootstrap_adaptation_rate * max(0.0, target - state["bootstrap_progress"])
            _apply_bootstrap_schedule(env, state, bootstrap_tracking_term, bootstrap_reward_term,
                                      bootstrap_velocity_floor_start, bootstrap_velocity_floor_end)
        gait_level = 0
        if gait_curriculum_state_attr is not None:
            gait_curriculum_state = getattr(env, gait_curriculum_state_attr, None)
            gait_level = -1 if gait_curriculum_state is None else int(gait_curriculum_state["level"])
        gait_level_ready = gait_level >= required_gait_level
        survival_rate = state["survival_sum"] / max(state["episode_count"], 1)
        stable_fraction = state["stable_samples"] / max(state["standing_samples"], 1.0)
        stopping_success = state["stopping_successes"] / max(state["stopping_attempts"], 1.0)
        directions = [0.0, 0.0, 0.0]

        global_promote = (
            standing_score >= promote_standing_score
            and gait_score >= promote_gait_score
            and gait_level_ready
            and survival_rate >= promote_survival_rate
        )
        global_demote = (
            standing_score <= demote_standing_score
            or gait_score <= demote_gait_score
            or not gait_level_ready
            or survival_rate <= demote_survival_rate
        )
        if stages is not None:
            global_promote &= stable_fraction >= promote_stable_fraction
            global_promote &= stopping_success >= promote_stopping_success
            global_demote |= stable_fraction < 0.5 or stopping_success < 0.5
        if physical_gait_term_name is not None:
            global_promote &= gait_score >= 1.0 - 1e-6
            global_demote |= gait_score <= 0.4
        promote_ready = global_promote and all(
            score >= promote_tracking_score for score, active in zip(tracking_scores, active_axes) if active
        )
        demote_ready = global_demote or any(
            score <= demote_tracking_score for score, active in zip(tracking_scores, active_axes) if active
        )
        stage_changed = False
        if stages is not None:
            state["good_windows"] = state["good_windows"] + 1 if promote_ready else 0
            state["bad_windows"] = state["bad_windows"] + 1 if demote_ready else 0
            previous_stage = state["stage"]
            if state["good_windows"] >= stage_promote_windows and state["stage"] < len(stages) - 1:
                state["stage"] += 1
            elif state["bad_windows"] >= stage_demote_windows and state["stage"] > 0:
                state["stage"] -= 1
            stage_changed = state["stage"] != previous_stage
            if stage_changed:
                state["good_windows"] = state["bad_windows"] = 0
                _apply_locomotion_stage(env, state, stages, stage_reward_terms, stage_push_term_name, physical_gait_term_name)
        for axis, name in enumerate(range_names):
            # Early stages keep their explicit low-speed envelopes. Only the
            # final stage expands speed, after all of its active axes pass.
            if stages is not None and (stage_changed or state["stage"] < len(stages) - 1):
                continue
            current_low, current_high = (float(v) for v in getattr(ranges, name))
            initial_low, initial_high = state["initial_ranges"][axis]
            limit_low, limit_high = (float(v) for v in getattr(limit_ranges, name))
            step = float(range_steps[axis])
            can_promote = global_promote if stages is None else promote_ready
            if can_promote and tracking_scores[axis] >= promote_tracking_score:
                new_low = max(limit_low, current_low - step)
                new_high = min(limit_high, current_high + step)
                directions[axis] = float(new_low != current_low or new_high != current_high)
            elif global_demote or tracking_scores[axis] <= demote_tracking_score:
                new_low = min(initial_low, current_low + step)
                new_high = max(initial_high, current_high - step)
                directions[axis] = -float(new_low != current_low or new_high != current_high)
            else:
                new_low, new_high = current_low, current_high
            setattr(ranges, name, [new_low, new_high])

        overall_direction = min(directions) if any(value < 0.0 for value in directions) else max(directions)
        state.update(
            {
                "axis_error_sum": [0.0, 0.0, 0.0],
                "axis_samples": [0.0, 0.0, 0.0],
                "standing_error_sum": 0.0,
                "standing_samples": 0.0,
                "stable_samples": 0.0,
                "stopping_attempts": 0.0,
                "stopping_successes": 0.0,
                "walking_seconds": 0.0,
                "walking_totals": [0.0] * 6,
                "gait_score_sum": 0.0,
                "gait_episode_count": 0,
                "survival_sum": 0.0,
                "episode_count": 0,
                "last_update_step": int(env.common_step_counter),
                "tracking_scores": tracking_scores,
                "standing_score": standing_score,
                "stable_fraction": stable_fraction,
                "stopping_success": stopping_success,
                "gait_score": gait_score,
                "survival_rate": survival_rate,
                "directions": directions,
                "direction": overall_direction,
                "promote_ready": promote_ready,
                "demote_ready": demote_ready,
            }
        )

    result = {
        "tracking_x": torch.tensor(state["tracking_scores"][0], device=env.device),
        "tracking_y": torch.tensor(state["tracking_scores"][1], device=env.device),
        "tracking_yaw": torch.tensor(state["tracking_scores"][2], device=env.device),
        "standing_score": torch.tensor(state["standing_score"], device=env.device),
        "gait_score": torch.tensor(state["gait_score"], device=env.device),
        "gait_level": torch.tensor(
            float(
                getattr(env, gait_curriculum_state_attr, {"level": 0})["level"]
                if gait_curriculum_state_attr is not None
                else 0
            ),
            device=env.device,
        ),
        "survival_rate": torch.tensor(state["survival_rate"], device=env.device),
        "direction_x": torch.tensor(state["directions"][0], device=env.device),
        "direction_y": torch.tensor(state["directions"][1], device=env.device),
        "direction_yaw": torch.tensor(state["directions"][2], device=env.device),
        "promote_ready": torch.tensor(float(state["promote_ready"]), device=env.device),
        "demote_ready": torch.tensor(float(state["demote_ready"]), device=env.device),
    }
    for axis, name in enumerate(range_names):
        low, high = getattr(ranges, name)
        result[f"{name}_lower"] = torch.tensor(low, device=env.device)
        result[f"{name}_upper"] = torch.tensor(high, device=env.device)
    if stages is not None:
        stage = stages[state["stage"]]
        for name, value in {
            "stage": state["stage"], "good_windows": state["good_windows"], "bad_windows": state["bad_windows"],
            "standing_fraction": stage["standing_fraction"], "lift_reward_scale": stage["lift_reward_scale"],
            "standing_stable_fraction": state["stable_fraction"], "stopping_success": state["stopping_success"],
        }.items():
            result[name] = torch.tensor(float(value), device=env.device)
    if physical_gait_term_name is not None:
        for name, value in {
            "left_step_rate_hz": state["walking_rates"][0], "right_step_rate_hz": state["walking_rates"][1],
            "left_mean_step_peak_m": state["walking_heights"][0], "right_mean_step_peak_m": state["walking_heights"][1],
            "walking_alternation": state["walking_alternation"],
            "evaluated_walking_seconds": state["evaluated_walking_seconds"],
            "evaluated_walking_transitions": state["evaluated_walking_transitions"],
        }.items():
            result[name] = torch.tensor(float(value), device=env.device)
    if bootstrap_schedule:
        result["bootstrap_progress"] = torch.tensor(state["bootstrap_progress"], device=env.device)
        result["velocity_reward_floor"] = torch.tensor(state["velocity_reward_floor"], device=env.device)
        result["bootstrap_reward_weight"] = torch.tensor(
            state["bootstrap_base_weight"] * (1.0 - state["bootstrap_progress"]), device=env.device
        )
    return result


def push_robot_velocity_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    event_term_name: str = "push_robot",
    start_x: float = 0.05,
    start_y: float = 0.03,
    end_x: float = 0.50,
    end_y: float = 0.30,
    start_yaw: float = 0.10,
    end_yaw: float = 0.30,
    step_x: float = 0.02,
    step_y: float = 0.01,
    step_yaw: float = 0.02,
) -> dict[str, torch.Tensor]:
    """Adapt push disturbance in the same direction as the performance curriculum."""
    if not hasattr(env, "_push_velocity_curriculum_state"):
        env._push_velocity_curriculum_state = {
            "x_vel": start_x,
            "y_vel": start_y,
            "yaw_vel": start_yaw,
            "last_performance_update_step": -1,
            "direction": 0.0,
        }
    state = env._push_velocity_curriculum_state

    performance_state = getattr(
        env,
        "_balanced_omni_curriculum_state",
        getattr(env, "_omni_vel_curriculum_state", None),
    )
    if performance_state is None:
        performance_state = getattr(env, "_lin_vel_curriculum_state", None)
    if performance_state is not None:
        performance_update_step = performance_state["last_update_step"]
        if performance_update_step != state["last_performance_update_step"]:
            promote_ready = performance_state.get(
                "promote_ready", performance_state["direction"] > 0.0
            )
            demote_ready = performance_state.get(
                "demote_ready", performance_state["direction"] < 0.0
            )
            if promote_ready:
                direction = 1.0
                state["x_vel"] = min(end_x, state["x_vel"] + step_x)
                state["y_vel"] = min(end_y, state["y_vel"] + step_y)
                state["yaw_vel"] = min(end_yaw, state["yaw_vel"] + step_yaw)
            elif demote_ready:
                direction = -1.0
                state["x_vel"] = max(start_x, state["x_vel"] - step_x)
                state["y_vel"] = max(start_y, state["y_vel"] - step_y)
                state["yaw_vel"] = max(start_yaw, state["yaw_vel"] - step_yaw)
            else:
                direction = 0.0
            state["last_performance_update_step"] = performance_update_step
            state["direction"] = direction

    x_vel = state["x_vel"]
    y_vel = state["y_vel"]
    yaw_vel = state["yaw_vel"]

    event_term = env.event_manager.get_term_cfg(event_term_name)
    event_term.params["velocity_range"]["x"] = (-x_vel, x_vel)
    event_term.params["velocity_range"]["y"] = (-y_vel, y_vel)
    event_term.params["velocity_range"]["yaw"] = (-yaw_vel, yaw_vel)

    return {
        "x": torch.tensor(x_vel, device=env.device),
        "y": torch.tensor(y_vel, device=env.device),
        "yaw": torch.tensor(yaw_vel, device=env.device),
        "direction": torch.tensor(state["direction"], device=env.device),
    }
