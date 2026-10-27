from isaaclab.utils import configclass

from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg


@configclass
class PPPORunnerCfg(BasePPORunnerCfg):
    experiment_name = "unitree_p_g1_rewards_pace"
    run_name = "official_g1_rewards_p_curriculum"
    clip_actions = 1.0  # Match P's bounded position-offset action and deployment.
