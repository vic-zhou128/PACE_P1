"""RSL-RL runner selection with optional bounds on learned action noise."""

import math

import torch
from rsl_rl.runners import OnPolicyRunner


class BoundedNoiseOnPolicyRunner(OnPolicyRunner):
    """Clamp noise parameters after each optimizer step and checkpoint load.

    The actor/critic architecture and state-dict keys remain standard RSL-RL,
    so deterministic inference and policy exports use the usual network.
    """

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.min_noise_std = float(train_cfg["min_noise_std"])
        self.max_noise_std = float(train_cfg["max_noise_std"])
        if not 0 < self.min_noise_std <= self.max_noise_std < float("inf"):
            raise ValueError("Expected finite 0 < min_noise_std <= max_noise_std")
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        if self.alg.policy.state_dependent_std:
            raise ValueError("BoundedNoiseOnPolicyRunner requires state-independent action noise")
        self._clamp_noise_parameters()
        self._noise_step_hook = self.alg.optimizer.register_step_post_hook(self._after_optimizer_step)
        self._noise_load_hook = self.alg.policy.register_load_state_dict_post_hook(self._after_policy_load)

    @torch.no_grad()
    def _clamp_noise_parameters(self):
        policy = self.alg.policy
        if policy.noise_std_type == "log":
            policy.log_std.clamp_(math.log(self.min_noise_std), math.log(self.max_noise_std))
        elif policy.noise_std_type == "scalar":
            policy.std.clamp_(self.min_noise_std, self.max_noise_std)
        else:
            raise ValueError(f"Unsupported action noise type: {policy.noise_std_type}")

    def _after_optimizer_step(self, optimizer, args, kwargs):
        self._clamp_noise_parameters()

    def _after_policy_load(self, module, incompatible_keys):
        self._clamp_noise_parameters()

    def save(self, path, infos=None):
        # Save the actual stage envelope with each policy, not just the global
        # deployment limits or the initial ranges in params/env.yaml.
        raw_env = getattr(self.env, "unwrapped", None)
        if raw_env is not None and hasattr(raw_env, "command_manager"):
            command = raw_env.command_manager.get_term("base_velocity")
            if hasattr(command.cfg, "limit_ranges"):
                infos = dict(infos or {})
                infos["p1_command_ranges"] = {
                    axis: tuple(getattr(command.cfg.ranges, axis))
                    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z")
                }
                state = getattr(raw_env, "_balanced_omni_curriculum_state", {})
                infos["p1_curriculum_stage"] = state.get("stage", 0)
        if hasattr(self, "finetune_source"):
            infos = dict(infos or {})
            infos["finetune_source"] = self.finetune_source
        super().save(path, infos)

    def log(self, locs, width=80, pad=35):
        super().log(locs, width=width, pad=pad)
        std = self.alg.policy.action_std.detach()
        self.writer.add_scalar("Policy/min_noise_std", std.min().item(), locs["it"])
        self.writer.add_scalar("Policy/max_noise_std", std.max().item(), locs["it"])
        clip = self.cfg.get("clip_actions")
        if clip is not None:
            # PPO.clear() resets the storage index, not the rollout tensors.
            fraction = (self.alg.storage.actions.abs() >= float(clip)).float().mean()
            self.writer.add_scalar("Policy/action_clip_fraction", fraction.item(), locs["it"])


def make_on_policy_runner(env, train_cfg, log_dir=None, device="cpu"):
    """Keep the standard runner unless the task explicitly configures noise bounds."""
    has_min = train_cfg.get("min_noise_std") is not None
    has_max = train_cfg.get("max_noise_std") is not None
    if has_min != has_max:
        raise ValueError("Configure both min_noise_std and max_noise_std")
    runner_type = BoundedNoiseOnPolicyRunner if has_min else OnPolicyRunner
    return runner_type(env, train_cfg, log_dir=log_dir, device=device)


def initialize_policy_for_finetuning(runner, checkpoint, noise_std=None):
    """Load policy/value weights into a fresh optimizer and fresh run counter."""
    if runner.alg.optimizer.state:
        raise ValueError("Fine-tuning initialization requires a fresh runner/optimizer")
    if noise_std is not None:
        if not math.isfinite(noise_std) or not 0 < noise_std:
            raise ValueError("Initial noise std must be finite and positive")
        if not getattr(runner, "min_noise_std", 0.0) <= noise_std <= getattr(runner, "max_noise_std", float("inf")):
            raise ValueError("Initial noise std is outside the task's exploration bounds")
    runner.load(checkpoint, load_optimizer=False)
    source_iteration = runner.current_learning_iteration
    runner.current_learning_iteration = 0
    runner.tot_timesteps = 0
    runner.tot_time = 0
    if noise_std is not None:
        policy = runner.alg.policy
        with torch.no_grad():
            if policy.noise_std_type == "log":
                policy.log_std.fill_(math.log(noise_std))
            elif policy.noise_std_type == "scalar":
                policy.std.fill_(noise_std)
            else:
                raise ValueError("Fine-tuning noise initialization requires a scalar/log noise parameter")
    runner.finetune_source = {"checkpoint": str(checkpoint), "iteration": int(source_iteration)}
    return {
        **runner.finetune_source, "reset_optimizer": True, "reset_iteration": True, "noise_std": noise_std,
        "learning_rate": runner.alg.optimizer.param_groups[0]["lr"],
        "schedule": runner.alg.schedule,
    }
