"""Separate experiment directory for measured lift and smoothness training."""

from isaaclab.utils import configclass

from .rsl_rl_dense_lift_ppo_cfg import DenseLiftPPORunnerCfg


@configclass
class TrueLiftSmoothPPORunnerCfg(DenseLiftPPORunnerCfg):
    experiment_name = "unitree_p_g1_pace_walk_true_lift_smooth"
    run_name = "alternating_steps_bounded_noise_from_scratch"
    min_noise_std: float = 0.05
    max_noise_std: float = 0.35

    def __post_init__(self):
        super().__post_init__()
        # Retain exploration within the explicit bounds enforced by our runner.
        self.algorithm.entropy_coef = 0.001


@configclass
class KneeLiftPPORunnerCfg(TrueLiftSmoothPPORunnerCfg):
    experiment_name = "unitree_p_g1_pace_walk_knee_lift"
    run_name = "knee_lift_continuous_steps_from_scratch"
