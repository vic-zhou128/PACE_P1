"""Fresh PPO run for forward walking before omnidirectional expansion."""

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import PPPORunnerCfg


@configclass
class PWalkPPORunnerCfg(PPPORunnerCfg):
    experiment_name = "unitree_p_g1_rewards_pace_walk"
    run_name = "walking_first"

    def __post_init__(self):
        self.policy.init_noise_std = 0.35
        self.policy.noise_std_type = "log"
        self.algorithm.entropy_coef = 0.0
