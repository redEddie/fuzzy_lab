import gymnasium as gym

from . import agents, flat_env_cfg, rough_env_cfg
from . import pothole_env_cfg, cylinder_env_cfg, fuzzy_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Velocity-Flat-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Flat-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2FlatPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2RoughPPORunnerCfg",
    },
)

# ------------------------

gym.register(
    id="Velocity-Pothole-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pothole_env_cfg.Go2PotholeEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PotholePPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Pothole-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": pothole_env_cfg.Go2PotholeEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2PotholePPORunnerCfg",
    },
)

# ------------------------

gym.register(
    id="Velocity-Cylinder-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cylinder_env_cfg.Go2CylinderEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2CylinderPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Cylinder-Go2-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": cylinder_env_cfg.Go2CylinderEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2CylinderPPORunnerCfg",
    },
)

# PPO + ANFIS Corrector environments
gym.register(
    id="Velocity-Flat-Go2-PPOANFISCorrector-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_anfis_corrector_cfg:PPOANFISCorrectorFlatRunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Go2-PPOANFISCorrector-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": rough_env_cfg.Go2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.ppo_anfis_corrector_cfg:PPOANFISCorrectorRoughRunnerCfg",
    },
)


# Hierarchical ANFIS environments
gym.register(
    id="Velocity-Flat-Go2-HierarchicalANFIS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fuzzy_env_cfg.Go2FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.hierarchical_anfis_ppo_cfg:Go2FlatHierarchicalANFISPPORunnerCfg",
    },
)

gym.register(
    id="Velocity-Rough-Go2-HierarchicalANFIS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": fuzzy_env_cfg.Go2RoughEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.hierarchical_anfis_ppo_cfg:Go2RoughHierarchicalANFISPPORunnerCfg",
    },
)

