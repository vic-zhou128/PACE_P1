"""Separate PPO logs for the 15-frame, no-phase PACE walking task."""

from isaaclab.utils import configclass

from .rsl_rl_g1_walk_ppo_cfg import G1WalkPPORunnerCfg


@configclass
class NoPhase15WalkPPORunnerCfg(G1WalkPPORunnerCfg):
    experiment_name = "unitree_p_g1_pace_walk_no_phase_15"
    run_name = "no_phase_15_from_scratch"
