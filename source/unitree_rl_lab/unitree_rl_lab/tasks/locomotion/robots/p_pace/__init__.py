import gymnasium as gym

gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth-KneeLift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KneeLiftEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:KneeLiftPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_true_lift_smooth_ppo_cfg:KneeLiftPPORunnerCfg",
    },
)

gym.register(
    id="Unitree-P-G1Rewards-PACE",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:PPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_ppo_cfg:PPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:WalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:WalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_walk_ppo_cfg:PWalkPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk-G1Events",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:G1WalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:G1WalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_g1_walk_ppo_cfg:G1WalkPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:NoPhase15WalkEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:NoPhase15WalkPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_no_phase_15_ppo_cfg:NoPhase15WalkPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-DenseLift",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DenseLiftEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:DenseLiftPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_dense_lift_ppo_cfg:DenseLiftPPORunnerCfg",
    },
)


gym.register(
    id="Unitree-P-G1Rewards-PACE-Walk-G1Events-NoPhase15-TrueLiftSmooth",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotEnvCfg",
        "play_env_cfg_entry_point": f"{__name__}.velocity_env_cfg:RobotPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_true_lift_smooth_ppo_cfg:TrueLiftSmoothPPORunnerCfg",
    },
)
