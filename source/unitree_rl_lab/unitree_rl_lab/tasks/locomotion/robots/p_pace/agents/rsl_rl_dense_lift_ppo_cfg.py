"""Separate logs for the dense 6 cm sole-clearance experiment."""

from isaaclab.utils import configclass

from .rsl_rl_no_phase_15_ppo_cfg import NoPhase15WalkPPORunnerCfg


@configclass
class DenseLiftPPORunnerCfg(NoPhase15WalkPPORunnerCfg):
    experiment_name = "unitree_p_g1_pace_walk_dense_lift"
    run_name = "dense_lift_from_scratch"
