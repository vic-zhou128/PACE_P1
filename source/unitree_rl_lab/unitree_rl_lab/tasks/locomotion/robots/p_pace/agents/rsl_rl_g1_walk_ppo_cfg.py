"""Separate logs for the G1-randomized P walking experiment."""

from isaaclab.utils import configclass

from .rsl_rl_walk_ppo_cfg import PWalkPPORunnerCfg


@configclass
class G1WalkPPORunnerCfg(PWalkPPORunnerCfg):
    experiment_name = "unitree_p_g1_pace_walk_g1_events"
    run_name = "g1_events_p_curriculum"
