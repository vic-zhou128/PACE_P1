"""Sole-clearance tracking for the P1 four-sphere feet on flat ground."""

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.utils.math import quat_apply


def _sole_clearance(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Lowest collision-sphere surface of each foot above the flat plane."""
    asset = env.scene[asset_cfg.name]
    positions = asset.data.body_pos_w[:, asset_cfg.body_ids]
    quaternions = asset.data.body_quat_w[:, asset_cfg.body_ids]
    centers = positions.new_tensor(
        [(-0.105, 0.045, -0.0328), (-0.105, -0.045, -0.0328),
         (0.065, 0.045, -0.0328), (0.065, -0.045, -0.0328)]
    )
    rotated = quat_apply(
        quaternions.unsqueeze(2).expand(-1, -1, 4, -1).reshape(-1, 4),
        centers.view(1, 1, 4, 3).expand(*positions.shape[:2], -1, -1).reshape(-1, 3),
    ).reshape(*positions.shape[:2], 4, 3)
    sole_height = positions[..., 2] + rotated[..., 2].amin(dim=-1) - 0.008
    return sole_height - env.scene.env_origins[:, 2].unsqueeze(1)


def _swing_arch(env, period: float, offset: tuple[float, float], threshold: float) -> torch.Tensor:
    phase = (env.episode_length_buf * env.step_dt / period).unsqueeze(1)
    phase = (phase + phase.new_tensor(offset)) % 1.0
    progress = ((phase - threshold) / (1.0 - threshold)).clamp(0.0, 1.0)
    return torch.sin(torch.pi * progress).square() * (phase >= threshold)


def swing_sole_clearance(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    period: float,
    offset: tuple[float, float],
    threshold: float,
    peak_clearance: float,
    height_std: float,
) -> torch.Tensor:
    """Track a smooth swing arch using the lowest foot collision sphere.

    Sphere centers/radius match urdf_803_nobattery/urdf/p1.urdf. Ground
    height is the environment origin's z, as required by this plane task.
    ``height_std`` is in meters and is squared in the exponential kernel.
    Weighting by the arch emphasizes mid-swing rather than near-ground motion.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    clearance = _sole_clearance(env, asset_cfg)
    arch = _swing_arch(env, period, offset, threshold)
    target = peak_clearance * arch
    tracking = torch.exp(-torch.square((clearance - target) / height_std))
    airborne = sensor.data.current_contact_time[:, sensor_cfg.body_ids] <= 0.0
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return (tracking * arch * airborne).sum(dim=1) * commanded


def swing_sole_lift_progress(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    period: float,
    offset: tuple[float, float],
    threshold: float,
    peak_clearance: float,
    height_floor: float,
) -> torch.Tensor:
    """Reward physical sole lift up to 6 cm, with no contact-sensor shortcut.

    The phase arch weights the middle of swing, while a fixed 6 cm denominator
    prevents a few millimeters of clearance from earning a large score near
    takeoff or landing. The floor removes resting contact jitter from the score.
    """
    clearance = _sole_clearance(env, asset_cfg)
    arch = _swing_arch(env, period, offset, threshold)
    lift_progress = ((clearance - height_floor) / (peak_clearance - height_floor)).clamp(0.0, 1.0)
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.1
    return (lift_progress * arch).sum(dim=1) * commanded


def _single_support_swing(env, sensor_cfg: SceneEntityCfg, max_air_time: float) -> torch.Tensor:
    """Select a briefly airborne foot while the opposite foot supports the body."""
    sensor = env.scene.sensors[sensor_cfg.name]
    contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    air_time = sensor.data.current_air_time[:, sensor_cfg.body_ids]
    one_contact = contact.sum(dim=1, keepdim=True) == 1
    return ~contact & one_contact & (air_time >= 0.04) & (air_time <= max_air_time)


def single_support_gait(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    min_clearance: float = 0.01,
    max_air_time: float = 0.6,
) -> torch.Tensor:
    """Reward real single-support steps without an unobserved phase clock.

    Both planted feet and both airborne feet earn zero. A held-up foot stops
    earning this reward after max_air_time, encouraging repeated touchdowns.
    """
    clearance = _sole_clearance(env, asset_cfg)
    swing = _single_support_swing(env, sensor_cfg, max_air_time)
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.05
    return ((clearance >= min_clearance) & swing).sum(dim=1).float() * commanded


def single_support_sole_lift_progress(
    env,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    peak_clearance: float,
    height_floor: float,
    max_air_time: float = 0.6,
) -> torch.Tensor:
    """Give dense 2 mm to 6 cm lift credit during physical single support."""
    clearance = _sole_clearance(env, asset_cfg)
    lift_progress = ((clearance - height_floor) / (peak_clearance - height_floor)).clamp(0.0, 1.0)
    swing = _single_support_swing(env, sensor_cfg, max_air_time)
    commanded = torch.linalg.norm(env.command_manager.get_command(command_name), dim=1) > 0.05
    return (lift_progress * swing).sum(dim=1) * commanded


class AlternatingFootstepReward(ManagerTermBase):
    """Own the shared physical-step state for all alternating-foot rewards.

    A step starts from confirmed ground support and ends at confirmed ground
    contact. After a rewarded landing only the opposite foot is eligible.
    Lift credit is a height-progress budget, not a per-frame holding bonus.
    """

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        p = cfg.params
        self.asset_cfg = p["asset_cfg"]
        self.sensor_cfg = p["sensor_cfg"]
        self.command_name = p["command_name"]
        self.takeoff_clearance = p.get("takeoff_clearance", 0.01)
        self.landing_clearance = p.get("landing_clearance", 0.005)
        self.min_step_height = p.get("min_step_height", 0.02)
        self.peak_clearance = p.get("peak_clearance", 0.06)
        self.height_floor = p.get("height_floor", 0.002)
        self.min_swing_time = p.get("min_swing_time", 0.12)
        self.max_swing_time = p.get("max_swing_time", 0.6)
        self.landing_time = p.get("landing_time", 0.06)
        self.min_step_interval = p.get("min_step_interval", 0.25)
        self.quality_window_s = p.get("quality_window_s", 4.0)
        self.quality_foot_rate_hz = p.get("quality_foot_rate_hz", 0.4)
        self.quality_height_m = p.get("quality_height_m", 0.025)
        if min(self.quality_window_s, self.quality_foot_rate_hz, self.quality_height_m) <= 0:
            raise ValueError("Walking quality window, foot rate and height must be positive")
        self.knee_joint_cfg = p.get("knee_joint_cfg")
        self.hip_roll_joint_cfg = p.get("hip_roll_joint_cfg")
        self.knee_flexion_start = p.get("knee_flexion_start", 0.08)
        self.knee_flexion_target = p.get("knee_flexion_target", 0.42)
        self.knee_separation_start = p.get("knee_separation_start", 0.04)
        self.knee_separation_target = p.get("knee_separation_target", 0.26)
        self.hip_roll_allowance = p.get("hip_roll_allowance", 0.08)
        self.hip_roll_scale = p.get("hip_roll_scale", 0.10)
        self.lateral_roll_allowance = p.get("lateral_roll_allowance", 0.08)
        self.yaw_roll_allowance = p.get("yaw_roll_allowance", 0.04)
        self.landing_knee_allowance = p.get("landing_knee_allowance", 0.16)
        self.landing_knee_scale = p.get("landing_knee_scale", 0.20)
        self.bootstrap_enabled = p.get("bootstrap_enabled", False)
        self.bootstrap_height_target = p.get("bootstrap_height_target", 0.025)
        self.bootstrap_min_height = p.get("bootstrap_min_height", 0.006)
        self.bootstrap_knee_start = p.get("bootstrap_knee_start", 0.01)
        self.bootstrap_knee_target = p.get("bootstrap_knee_target", 0.18)
        self.bootstrap_max_time = p.get("bootstrap_max_time", 0.8)
        self.bootstrap_min_unload_time = p.get("bootstrap_min_unload_time", 0.06)
        if self.bootstrap_enabled and self.knee_joint_cfg is None:
            raise ValueError("Bootstrap guidance requires knee and hip-roll joints")
        if self.bootstrap_enabled and not self.height_floor < self.bootstrap_min_height <= self.bootstrap_height_target:
            raise ValueError("Invalid bootstrap height thresholds")
        if self.bootstrap_enabled and (not 0 <= self.bootstrap_knee_start < self.bootstrap_knee_target or not 0 < self.bootstrap_min_unload_time < self.bootstrap_max_time):
            raise ValueError("Invalid bootstrap knee or timing thresholds")
        if not (
            0 <= self.height_floor < self.landing_clearance < self.takeoff_clearance
            <= self.min_step_height <= self.peak_clearance
        ):
            raise ValueError("Expected floor < landing < takeoff <= minimum step height <= target clearance")
        if not 0 < self.min_swing_time <= self.max_swing_time or self.landing_time <= 0 or self.min_step_interval < 0:
            raise ValueError("Invalid swing, landing or step interval timings")
        if len(self.asset_cfg.body_ids) != 2 or len(self.sensor_cfg.body_ids) != 2:
            raise ValueError("Alternating footsteps require explicit left/right foot ordering")
        if (self.knee_joint_cfg is None) != (self.hip_roll_joint_cfg is None):
            raise ValueError("Configure both knee and hip-roll joints for knee shaping")
        if self.knee_joint_cfg is not None:
            if len(self.knee_joint_cfg.joint_ids) != 2 or len(self.hip_roll_joint_cfg.joint_ids) != 2:
                raise ValueError("Knee shaping requires explicit left/right joint ordering")
            if self.knee_joint_cfg.name != self.asset_cfg.name or self.hip_roll_joint_cfg.name != self.asset_cfg.name:
                raise ValueError("Knee, hip-roll and foot entities must refer to the same robot")
            if not 0 <= self.knee_flexion_start < self.knee_flexion_target:
                raise ValueError("Expected 0 <= knee_flexion_start < knee_flexion_target")
            if not 0 <= self.knee_separation_start < self.knee_separation_target:
                raise ValueError("Expected 0 <= knee_separation_start < knee_separation_target")
            if min(self.hip_roll_scale, self.landing_knee_scale) <= 0:
                raise ValueError("Joint penalty scales must be positive")
            if min(self.hip_roll_allowance, self.lateral_roll_allowance,
                   self.yaw_roll_allowance, self.landing_knee_allowance) < 0:
                raise ValueError("Joint allowances must be nonnegative")
        shape = (env.num_envs, 2)
        self.active = torch.zeros(shape, dtype=torch.bool, device=env.device)
        self.armed = torch.zeros_like(self.active)
        self.contact_seen = torch.zeros_like(self.active)
        self.invalid = torch.zeros_like(self.active)
        self.ground_time = torch.zeros(shape, device=env.device)
        self.swing_time = torch.zeros_like(self.ground_time)
        self.support_time = torch.zeros_like(self.ground_time)
        self.peak = torch.zeros_like(self.ground_time)
        self.lift_credit = torch.zeros_like(self.ground_time)
        self.valid_steps = torch.zeros_like(self.ground_time)
        self.rewarded_steps = torch.zeros_like(self.ground_time)
        self.clearance_peak = torch.zeros_like(self.ground_time)
        self.required_foot = torch.full((env.num_envs,), -1, dtype=torch.long, device=env.device)
        self.last_physical_foot = torch.full_like(self.required_foot, -1)
        self.transitions = torch.zeros(env.num_envs, device=env.device)
        self.alternations = torch.zeros_like(self.transitions)
        self.repeated_steps = torch.zeros_like(self.transitions)
        self.since_rewarded_step = torch.full_like(self.transitions, self.min_step_interval)
        self.completion_credit = torch.zeros_like(self.transitions)
        self.new_lift_credit = torch.zeros_like(self.transitions)
        self.completed_air_time = torch.zeros_like(self.transitions)
        self.knee_credit = torch.zeros_like(self.ground_time)
        self.new_knee_credit = torch.zeros_like(self.transitions)
        self.swing_roll_cost = torch.zeros_like(self.transitions)
        self.landing_knee_cost = torch.zeros_like(self.transitions)
        self.knee_at_peak = torch.zeros_like(self.ground_time)
        self.knee_separation_at_peak = torch.zeros_like(self.ground_time)
        self.roll_at_peak = torch.zeros_like(self.ground_time)
        self.body_roll_at_peak = torch.zeros_like(self.ground_time)
        self.knee_peak_sum = torch.zeros_like(self.ground_time)
        self.knee_separation_sum = torch.zeros_like(self.ground_time)
        self.roll_peak_sum = torch.zeros_like(self.ground_time)
        self.body_roll_peak_sum = torch.zeros_like(self.ground_time)
        self.knee_flexed_steps = torch.zeros_like(self.ground_time)
        # Planar walking only: [left/right count, left/right peak sum, transitions, alternations].
        # A fixed-duration ring prevents a single initial pair from qualifying a
        # long shuffle. Standing clears recent evidence, but not episode totals.
        self.walking_time = torch.zeros(env.num_envs, device=env.device)
        self.walking_totals = torch.zeros(env.num_envs, 6, device=env.device)
        self.walking_quality_sum = torch.zeros_like(self.walking_time)
        self.recent_quality = torch.zeros_like(self.walking_time)
        self._quality_steps = max(1, round(self.quality_window_s / env.step_dt))
        self._quality_duration = self._quality_steps * env.step_dt
        self._quality_history = torch.zeros(self._quality_steps, env.num_envs, 6, device=env.device)
        self._quality_sums = torch.zeros_like(self.walking_totals)
        self._quality_cursor = 0
        self._walking_last_foot = torch.full_like(self.required_foot, -1)
        self._was_walking = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.bootstrap_active = torch.zeros_like(self.active)
        self.bootstrap_armed = torch.zeros_like(self.active)
        self.bootstrap_invalid = torch.zeros_like(self.active)
        self.bootstrap_time = torch.zeros_like(self.ground_time)
        self.bootstrap_unload_time = torch.zeros_like(self.ground_time)
        self.bootstrap_peak_height = torch.zeros_like(self.ground_time)
        self.bootstrap_peak_knee = torch.zeros_like(self.ground_time)
        self.bootstrap_peak_separation = torch.zeros_like(self.ground_time)
        self.bootstrap_credit = torch.zeros_like(self.ground_time)
        self.bootstrap_steps = torch.zeros_like(self.ground_time)
        self.bootstrap_paid = torch.zeros_like(self.ground_time)
        self.bootstrap_required_foot = torch.full_like(self.required_foot, -1)
        self.new_bootstrap_credit = torch.zeros_like(self.walking_time)
        self._last_update = -1

    def update(self):
        env = self._env
        if self._last_update == env.common_step_counter:
            return
        self._last_update = env.common_step_counter
        dt = env.step_dt
        height = _sole_clearance(env, self.asset_cfg)
        sensor = env.scene.sensors[self.sensor_cfg.name]
        contact = sensor.data.current_contact_time[:, self.sensor_cfg.body_ids] > 0.0
        command = env.command_manager.get_command(self.command_name)
        moving = torch.linalg.norm(command, dim=1) > 0.05
        walking = torch.linalg.norm(command[:, :2], dim=1) > 0.05
        self.walking_time += walking * dt
        sample = self._quality_history[self._quality_cursor]
        self._quality_sums -= sample
        sample.zero_()
        stopped = self._was_walking & ~walking
        self._quality_history[:, stopped] = 0.0
        self._quality_sums[stopped] = 0.0
        self._walking_last_foot[~walking] = -1
        self._was_walking.copy_(walking)
        self.clearance_peak = torch.maximum(self.clearance_peak, height)
        self.since_rewarded_step += dt
        self.completion_credit.zero_()
        self.new_lift_credit.zero_()
        self.completed_air_time.zero_()
        self.new_knee_credit.zero_()
        self.swing_roll_cost.zero_()
        self.landing_knee_cost.zero_()
        if self.knee_joint_cfg is not None:
            robot = env.scene[self.asset_cfg.name]
            knee_delta = (
                robot.data.joint_pos[:, self.knee_joint_cfg.joint_ids]
                - robot.data.default_joint_pos[:, self.knee_joint_cfg.joint_ids]
            )
            knee_separation = knee_delta - knee_delta.flip(dims=(1,))
            roll_delta = (
                robot.data.joint_pos[:, self.hip_roll_joint_cfg.joint_ids]
                - robot.data.default_joint_pos[:, self.hip_roll_joint_cfg.joint_ids]
            ).abs()
            gravity = robot.data.projected_gravity_b
            body_roll = torch.atan2(-gravity[:, 1], -gravity[:, 2]).abs().unsqueeze(1)
            roll_allowance = (
                self.hip_roll_allowance
                + self.lateral_roll_allowance * (command[:, 1].abs() / 0.15).clamp(max=1.0)
                + self.yaw_roll_allowance * (command[:, 2].abs() / 0.5).clamp(max=1.0)
            ).unsqueeze(1)
            roll_excess = ((roll_delta - roll_allowance).clamp_min(0.0) / self.hip_roll_scale).square()

        low = height <= self.landing_clearance
        self.ground_time = torch.where(low, self.ground_time + dt, 0.0)
        self.contact_seen = low & (self.contact_seen | contact)
        grounded = (self.ground_time >= self.landing_time - 1.0e-6) & self.contact_seen
        opposite_support = grounded.flip(dims=(1,))
        self.active &= moving.unsqueeze(1)
        self.armed |= grounded & ~self.active
        high = height >= self.takeoff_clearance
        started = ~self.active & self.armed & high & opposite_support & moving.unsqueeze(1)
        self.armed &= ~high
        self.swing_time[started] = 0.0
        self.support_time[started] = 0.0
        self.peak[started] = 0.0
        self.invalid[started] = False
        self.active |= started
        self.swing_time += self.active * dt
        self.support_time += (self.active & opposite_support) * dt
        if self.knee_joint_cfg is not None:
            higher = self.active & ((height > self.peak) | started)
            self.knee_at_peak = torch.where(higher, knee_delta, self.knee_at_peak)
            self.knee_separation_at_peak = torch.where(higher, knee_separation, self.knee_separation_at_peak)
            self.roll_at_peak = torch.where(higher, roll_delta, self.roll_at_peak)
            self.body_roll_at_peak = torch.where(higher, body_roll, self.body_roll_at_peak)
        self.peak = torch.where(self.active, torch.maximum(self.peak, height), self.peak)
        self.invalid |= self.active & high.all(dim=1, keepdim=True)

        landed = self.active & grounded
        air_duration = (self.swing_time - self.ground_time).clamp_min(0.0)
        completed = landed & ~self.invalid & (self.peak >= self.min_step_height)
        completed &= (air_duration >= self.min_swing_time - 1.0e-6)
        completed &= (air_duration <= self.max_swing_time + 1.0e-6)
        completed &= (self.support_time >= self.min_swing_time - 1.0e-6) & moving.unsqueeze(1)
        self.valid_steps += completed
        if self.knee_joint_cfg is not None:
            self.knee_peak_sum += completed * self.knee_at_peak
            self.knee_separation_sum += completed * self.knee_separation_at_peak
            self.roll_peak_sum += completed * self.roll_at_peak
            self.body_roll_peak_sum += completed * self.body_roll_at_peak
            flexed = self.knee_at_peak >= self.knee_flexion_target
            flexed &= self.knee_separation_at_peak >= self.knee_separation_target
            self.knee_flexed_steps += completed & flexed
            landing_excess = (
                (knee_delta - self.landing_knee_allowance).clamp_min(0.0) / self.landing_knee_scale
            ).square().clamp(max=4.0)
            self.landing_knee_cost = (completed * landing_excess).sum(dim=1)
        for foot in range(2):
            event = completed[:, foot]
            walking_event = event & walking
            previous_walking_step = self._walking_last_foot >= 0
            sample[:, foot] = walking_event
            sample[:, 2 + foot] = walking_event * self.peak[:, foot]
            sample[:, 4] += walking_event & previous_walking_step
            sample[:, 5] += walking_event & previous_walking_step & (self._walking_last_foot != foot)
            self._walking_last_foot[walking_event] = foot
            has_previous = self.last_physical_foot >= 0
            self.transitions += event & has_previous
            self.alternations += event & has_previous & (self.last_physical_foot != foot)
            self.repeated_steps += event & has_previous & (self.last_physical_foot == foot)
            self.last_physical_foot[event] = foot
            accepted = event & ((self.required_foot == -1) | (self.required_foot == foot))
            accepted &= self.since_rewarded_step >= self.min_step_interval - 1.0e-6
            peak_progress = (
                (self.peak[:, foot] - self.height_floor) / (self.peak_clearance - self.height_floor)
            ).clamp(0.0, 1.0)
            self.completion_credit += accepted * peak_progress
            self.completed_air_time += accepted * air_duration[:, foot].clamp(max=0.4)
            self.rewarded_steps[:, foot] += accepted
            self.required_foot[accepted] = 1 - foot
            self.since_rewarded_step[accepted] = 0.0
            self.lift_credit[accepted, 1 - foot] = 0.0
            self.knee_credit[accepted, 1 - foot] = 0.0
        self.active &= ~landed
        self.armed |= landed

        foot_ids = torch.arange(2, device=env.device).unsqueeze(0)
        eligible = (self.required_foot.unsqueeze(1) == -1) | (self.required_foot.unsqueeze(1) == foot_ids)
        # A brief sensor contact does not restart either this timer or the
        # height-credit budget. Retries of the same phase cannot repay credit.
        physical_swing = self.active & ~self.invalid & opposite_support
        physical_swing &= (self.swing_time <= self.max_swing_time + 1.0e-6) & moving.unsqueeze(1)
        valid_swing = physical_swing & eligible
        progress = ((height - self.height_floor) / (self.peak_clearance - self.height_floor)).clamp(0.0, 1.0)
        new_credit = torch.where(valid_swing, (progress - self.lift_credit).clamp_min(0.0), 0.0)
        self.lift_credit += new_credit
        self.new_lift_credit = new_credit.sum(dim=1)
        if self.knee_joint_cfg is not None:
            knee_progress = (
                (knee_delta - self.knee_flexion_start) / (self.knee_flexion_target - self.knee_flexion_start)
            ).clamp(0.0, 1.0)
            # A permanent two-knee crouch cannot substitute for swing flexion.
            separation_progress = (
                (knee_separation - self.knee_separation_start)
                / (self.knee_separation_target - self.knee_separation_start)
            ).clamp(0.0, 1.0)
            knee_progress = torch.minimum(knee_progress, separation_progress)
            # The bonus favors coordinated flexion with limited side swing.
            # Height and flexion alone do not establish causal lift contribution.
            score = progress * knee_progress * torch.exp(-roll_excess)
            knee_credit = torch.where(valid_swing, (score - self.knee_credit).clamp_min(0.0), 0.0)
            self.knee_credit += knee_credit
            self.new_knee_credit = knee_credit.sum(dim=1)
            # Same-leg retries still pay the side-swing cost; only bonuses alternate.
            self.swing_roll_cost = (physical_swing * roll_excess.clamp(max=4.0)).sum(dim=1)

        self.walking_totals += sample
        self._quality_sums.add_(sample).clamp_min_(0.0)
        counts = self._quality_sums[:, :2]
        rate_score = (counts.amin(dim=1) / (self._quality_duration * self.quality_foot_rate_hz)).clamp(0.0, 1.0)
        mean_heights = self._quality_sums[:, 2:4] / counts.clamp_min(1.0)
        height_score = (mean_heights.amin(dim=1) / self.quality_height_m).clamp(0.0, 1.0)
        alternation = self._quality_sums[:, 5] / self._quality_sums[:, 4].clamp_min(1.0)
        self.recent_quality = rate_score * height_score * alternation.clamp(0.0, 1.0) * walking
        self.walking_quality_sum += self.recent_quality * dt
        self._quality_cursor = (self._quality_cursor + 1) % self._quality_steps
        self.new_bootstrap_credit.zero_()
        if self.bootstrap_enabled:
            self._update_bootstrap(height, grounded, knee_delta, knee_separation, roll_excess, walking)

    def _update_bootstrap(self, height, grounded, knee_delta, knee_separation, roll_excess, walking):
        """Finite pre-takeoff credit; only alternate, flexed mini-landings renew it.

        This separate state never increments the >=2 cm physical gait counters.
        Equal-load planted crouches, held legs and stop/start retries cannot
        repeatedly collect a phase's credit.
        """
        dt = self._env.step_dt
        sensor = self._env.scene.sensors[self.sensor_cfg.name]
        force = sensor.data.net_forces_w_history[:, :, self.sensor_cfg.body_ids, :].norm(dim=-1).amax(dim=1)
        opposite_grounded = grounded.flip(dims=(1,))
        opposite_force = force.flip(dims=(1,))
        unloaded = (force < 0.5 * opposite_force) & (opposite_force > 1.0)
        allowed = walking[:, None] & opposite_grounded
        foot_ids = torch.arange(2, device=self._env.device)[None, :]
        eligible = (self.bootstrap_required_foot[:, None] == -1) | (self.bootstrap_required_foot[:, None] == foot_ids)
        self.bootstrap_active &= walking[:, None]
        self.bootstrap_armed |= grounded & ~self.bootstrap_active
        started = self.bootstrap_armed & ~self.bootstrap_active & allowed & unloaded & eligible
        self.bootstrap_active |= started
        self.bootstrap_armed &= ~started
        for value in (self.bootstrap_time, self.bootstrap_unload_time, self.bootstrap_peak_height,
                      self.bootstrap_peak_knee, self.bootstrap_peak_separation):
            value[started] = 0.0
        self.bootstrap_invalid[started] = False
        self.bootstrap_time += self.bootstrap_active * dt
        self.bootstrap_unload_time += (self.bootstrap_active & unloaded & opposite_grounded) * dt
        self.bootstrap_invalid |= self.bootstrap_active & ~opposite_grounded
        self.bootstrap_peak_height = torch.where(self.bootstrap_active, torch.maximum(self.bootstrap_peak_height, height), self.bootstrap_peak_height)
        self.bootstrap_peak_knee = torch.where(self.bootstrap_active & unloaded, torch.maximum(self.bootstrap_peak_knee, knee_delta), self.bootstrap_peak_knee)
        self.bootstrap_peak_separation = torch.where(self.bootstrap_active & unloaded, torch.maximum(self.bootstrap_peak_separation, knee_separation), self.bootstrap_peak_separation)
        valid = self.bootstrap_active & allowed & unloaded & ~self.bootstrap_invalid
        valid &= self.bootstrap_time <= self.bootstrap_max_time
        knee_progress = ((knee_delta - self.bootstrap_knee_start) / (self.bootstrap_knee_target - self.bootstrap_knee_start)).clamp(0.0, 1.0)
        separation = ((knee_separation - 0.01) / 0.11).clamp(0.0, 1.0)
        knee_progress = torch.minimum(knee_progress, separation)
        height_progress = ((height - self.height_floor) / (self.bootstrap_height_target - self.height_floor)).clamp(0.0, 1.0)
        score = (0.4 * knee_progress + 0.6 * height_progress) * torch.exp(-roll_excess)
        credit = torch.where(valid & eligible, (score - self.bootstrap_credit).clamp_min(0.0), 0.0)
        self.bootstrap_credit += credit
        self.bootstrap_paid += credit
        self.new_bootstrap_credit = credit.sum(dim=1)
        first = (self.bootstrap_required_foot == -1) & (self.new_bootstrap_credit > 0)
        self.bootstrap_required_foot[first] = credit.argmax(dim=1)[first]
        # 6 mm is a bootstrap landing only; the main step detector still needs
        # 2 cm, 120 ms air time and confirmed opposite support.
        landed = self.bootstrap_active & grounded & (self.bootstrap_peak_height >= self.bootstrap_min_height)
        accepted = landed & ~self.bootstrap_invalid & eligible
        accepted &= (self.bootstrap_unload_time >= self.bootstrap_min_unload_time - 1e-6) & (self.bootstrap_time <= self.bootstrap_max_time)
        accepted &= (self.bootstrap_peak_knee >= self.knee_flexion_start) & (self.bootstrap_peak_separation >= self.knee_separation_start)
        for foot in range(2):
            event = accepted[:, foot]
            self.bootstrap_steps[:, foot] += event
            self.bootstrap_required_foot[event] = 1 - foot
            self.bootstrap_credit[event, 1 - foot] = 0.0
        self.bootstrap_active &= ~landed & (self.bootstrap_time <= self.bootstrap_max_time)

    def __call__(
        self, env, asset_cfg, sensor_cfg, command_name,
        takeoff_clearance=0.01, landing_clearance=0.005, min_step_height=0.02,
        peak_clearance=0.06, height_floor=0.002, min_swing_time=0.12,
        max_swing_time=0.6, landing_time=0.06, min_step_interval=0.25,
        knee_joint_cfg=None, hip_roll_joint_cfg=None,
        knee_flexion_start=0.08, knee_flexion_target=0.42,
        knee_separation_start=0.04, knee_separation_target=0.26,
        hip_roll_allowance=0.08, hip_roll_scale=0.10,
        lateral_roll_allowance=0.08, yaw_roll_allowance=0.04,
        landing_knee_allowance=0.16, landing_knee_scale=0.20,
        quality_window_s=4.0, quality_foot_rate_hz=0.4, quality_height_m=0.025,
        bootstrap_enabled=False,
        bootstrap_height_target=0.025, bootstrap_min_height=0.006,
        bootstrap_knee_start=0.01, bootstrap_knee_target=0.18,
        bootstrap_max_time=0.8, bootstrap_min_unload_time=0.06,
    ):
        self.update()
        # RewardManager multiplies by dt: these are per-event credits.
        return self.completion_credit.clone() / env.step_dt

    def reset(self, env_ids=None):
        if env_ids is not None and not isinstance(env_ids, slice) and len(env_ids) == 0:
            return
        ids = slice(None) if env_ids is None else env_ids
        log = self._env.extras.setdefault("log", {})
        metrics = {
            "left_valid_steps": self.valid_steps[ids, 0],
            "right_valid_steps": self.valid_steps[ids, 1],
            "left_rewarded_steps": self.rewarded_steps[ids, 0],
            "right_rewarded_steps": self.rewarded_steps[ids, 1],
            "alternation_rate": self.alternations[ids] / self.transitions[ids].clamp_min(1.0),
            "step_transitions": self.transitions[ids],
            "same_leg_repeats": self.repeated_steps[ids],
            "left_clearance_peak_m": self.clearance_peak[ids, 0],
            "right_clearance_peak_m": self.clearance_peak[ids, 1],
            "walking_time_s": self.walking_time[ids],
            "left_walking_step_rate_hz": self.walking_totals[ids, 0] / self.walking_time[ids].clamp_min(1e-6),
            "right_walking_step_rate_hz": self.walking_totals[ids, 1] / self.walking_time[ids].clamp_min(1e-6),
            "left_walking_mean_step_peak_m": self.walking_totals[ids, 2] / self.walking_totals[ids, 0].clamp_min(1.0),
            "right_walking_mean_step_peak_m": self.walking_totals[ids, 3] / self.walking_totals[ids, 1].clamp_min(1.0),
            "walking_alternation_rate": self.walking_totals[ids, 5] / self.walking_totals[ids, 4].clamp_min(1.0),
            "walking_transitions": self.walking_totals[ids, 4],
            "mean_walking_quality": self.walking_quality_sum[ids] / self.walking_time[ids].clamp_min(1e-6),
            "left_bootstrap_steps": self.bootstrap_steps[ids, 0],
            "right_bootstrap_steps": self.bootstrap_steps[ids, 1],
            "bootstrap_credit_paid": self.bootstrap_paid[ids].sum(dim=1),
        }
        if self.knee_joint_cfg is not None:
            for foot, side in enumerate(("left", "right")):
                count = self.valid_steps[ids, foot].clamp_min(1.0)
                metrics[f"{side}_peak_knee_delta_rad"] = self.knee_peak_sum[ids, foot] / count
                metrics[f"{side}_peak_knee_separation_rad"] = self.knee_separation_sum[ids, foot] / count
                metrics[f"{side}_peak_hip_roll_abs_rad"] = self.roll_peak_sum[ids, foot] / count
                metrics[f"{side}_peak_body_roll_abs_rad"] = self.body_roll_peak_sum[ids, foot] / count
                metrics[f"{side}_knee_flexed_step_fraction"] = self.knee_flexed_steps[ids, foot] / count
        log.update({f"Metrics/gait/{name}": value.float().mean().clone() for name, value in metrics.items()})
        for value in (
            self.active, self.armed, self.contact_seen, self.invalid, self.ground_time,
            self.swing_time, self.support_time, self.peak, self.lift_credit, self.valid_steps,
            self.rewarded_steps, self.clearance_peak, self.transitions, self.alternations,
            self.repeated_steps, self.completion_credit, self.new_lift_credit, self.completed_air_time,
            self.knee_credit, self.new_knee_credit, self.swing_roll_cost, self.landing_knee_cost,
            self.knee_at_peak, self.knee_separation_at_peak, self.roll_at_peak, self.body_roll_at_peak,
            self.knee_peak_sum, self.knee_separation_sum,
            self.roll_peak_sum, self.body_roll_peak_sum, self.knee_flexed_steps,
            self.walking_time, self.walking_totals, self.walking_quality_sum, self.recent_quality,
            self._quality_sums, self._was_walking,
            self.bootstrap_active, self.bootstrap_armed, self.bootstrap_invalid,
            self.bootstrap_time, self.bootstrap_unload_time, self.bootstrap_peak_height,
            self.bootstrap_peak_knee, self.bootstrap_peak_separation, self.bootstrap_credit,
            self.bootstrap_steps, self.bootstrap_paid, self.new_bootstrap_credit,
        ):
            value[ids] = 0
        self.required_foot[ids] = -1
        self.last_physical_foot[ids] = -1
        self.since_rewarded_step[ids] = self.min_step_interval
        self._quality_history[:, ids] = 0.0
        self._walking_last_foot[ids] = -1
        self.bootstrap_required_foot[ids] = -1


def _alternating_state(env, gait_reward_name):
    state = env.reward_manager.get_term_cfg(gait_reward_name).func
    if not isinstance(state, AlternatingFootstepReward):
        raise TypeError("Alternating lift/air-time terms require an AlternatingFootstepReward gait term")
    state.update()
    return state


def alternating_sole_lift_credit(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _alternating_state(env, gait_reward_name).new_lift_credit.clone() / env.step_dt


def alternating_air_time_credit(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _alternating_state(env, gait_reward_name).completed_air_time.clone() / env.step_dt


def _knee_state(env, gait_reward_name):
    state = _alternating_state(env, gait_reward_name)
    if state.knee_joint_cfg is None:
        raise ValueError("Knee-shaping terms require knee_joint_cfg and hip_roll_joint_cfg in the gait term")
    return state


def alternating_knee_lift_credit(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _knee_state(env, gait_reward_name).new_knee_credit.clone() / env.step_dt


def bootstrap_knee_lift_credit(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _knee_state(env, gait_reward_name).new_bootstrap_credit.clone() / env.step_dt


def swing_hip_roll_excess(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _knee_state(env, gait_reward_name).swing_roll_cost.clone()


def landing_knee_excess(env, gait_reward_name: str = "gait") -> torch.Tensor:
    return _knee_state(env, gait_reward_name).landing_knee_cost.clone() / env.step_dt
