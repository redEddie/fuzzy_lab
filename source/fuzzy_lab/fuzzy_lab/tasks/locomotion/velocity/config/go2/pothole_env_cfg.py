from isaaclab.utils import configclass

from .rough_env_cfg import Go2RoughEnvCfg
import isaaclab.terrains as terrain_gen


@configclass
class Go2PotholeEnvCfg(Go2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        
        # Fix PhysX patch buffer overflow for complex terrain
        # Increase patch buffer size to handle collision complexity
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15  # Increase from default 10 * 2**15
        self.sim.physx.gpu_found_lost_pairs_capacity = 2**21   # Increase pair capacity
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**16  # Increase aggregate pairs
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 2**16       # Total aggregate pairs
        self.sim.physx.friction_correlation_distance = 0.025   # Optimize friction processing

        # override rewards
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.dof_torques_l2.weight = -2.5e-5
        self.rewards.feet_air_time.weight = 0.5
        # change terrain to flat
        # self.scene.terrain.terrain_type = "plane"
        # self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None

        self.scene.terrain.terrain_generator.sub_terrains={
            # 평지 지형
            "flat": terrain_gen.MeshPlaneTerrainCfg(
                proportion=0.3,  # 30% 비율로 평지
            ),
            # Stepping stones 지형
            "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
                proportion=0.3,  # 30% 비율
                stone_width_range=(0.3, 0.8),
                stone_height_max=0.1,
                stone_distance_range=(0.1, 0.3),
                platform_width=1.0,
            ),
            # Gap 지형
            "gaps": terrain_gen.MeshGapTerrainCfg(
                proportion=0.2,  # 20% 비율
                gap_width_range=(0.2, 0.8),
                platform_width=1.0,
            ),
            # 기존 지형들도 일부 유지 (비율 조정)
            "boxes": terrain_gen.MeshRandomGridTerrainCfg(
                proportion=0.1, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
            ),
            "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=0.1, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
            ),
        }


class Go2PotholeEnvCfg_PLAY(Go2PotholeEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
