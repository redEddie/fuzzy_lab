# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

# --------------

import isaaclab.terrains.trimesh.mesh_terrains as mesh_terrains
import isaaclab.terrains.trimesh.mesh_terrains_cfg as mesh_terrains_cfg
import trimesh
import numpy as np
import math

# --------

from isaaclab.utils import configclass
from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from dataclasses import MISSING
from collections.abc import Callable

def random_hole_grid_terrain(
    difficulty: float, cfg: "MeshRandomHoleGridTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """바둑판 형태로 나누어 랜덤하게 구멍을 뚫는 지형"""
    
    # 난이도에 따른 구멍 확률 계산
    hole_probability = cfg.hole_prob_range[0] + difficulty * (cfg.hole_prob_range[1] - cfg.hole_prob_range[0])
    
    meshes_list = []
    terrain_height = 1.0
    
    # 1. 바둑판 셀 계산 (random_grid_terrain 방식)
    num_cells_x = math.ceil(cfg.size[0] / cfg.cell_width)
    num_cells_y = math.ceil(cfg.size[1] / cfg.cell_width)
    
    # 2. 중앙 플랫폼 영역 정의
    center_x, center_y = num_cells_x // 2, num_cells_y // 2
    platform_size = cfg.platform_size  # 예: 2.0m × 2.0m
    platform_cells_x = int(platform_size / cfg.cell_width)
    platform_cells_y = int(platform_size / cfg.cell_width)

    center_x, center_y = num_cells_x // 2, num_cells_y // 2
    platform_half_x = platform_cells_x // 2
    platform_half_y = platform_cells_y // 2

    def is_in_platform_area(i, j, center_x, center_y, platform_half_x, platform_half_y):
        """중앙 플랫폼 영역 내부인지 확인"""
        return (center_x - platform_half_x <= i <= center_x + platform_half_x and
                center_y - platform_half_y <= j <= center_y + platform_half_y)
    
    # 3. 각 셀별로 구멍/지면 결정
    for i in range(num_cells_x):
        for j in range(num_cells_y):
            is_platform_area = is_in_platform_area(i, j, center_x, center_y, platform_half_x, platform_half_y)
            
            # 중앙이면 무조건 지면 생성, 아니면 확률적으로 결정
            if is_platform_area or np.random.random() > hole_probability:
                # 지면 셀 생성
                cell_pos = (
                    i * cfg.cell_width + cfg.cell_width/2,
                    j * cfg.cell_width + cfg.cell_width/2,
                    -terrain_height/2
                )
                cell_dims = (cfg.cell_width, cfg.cell_width, terrain_height)
                cell_mesh = trimesh.creation.box(
                    cell_dims, 
                    trimesh.transformations.translation_matrix(cell_pos)
                )
                meshes_list.append(cell_mesh)
            # else: 구멍 → 아무것도 생성하지 않음
    
    # 4. Origin은 중앙 플랫폼에
    origin = np.array([cfg.size[0]/2, cfg.size[1]/2, 0.0])
    
    return meshes_list, origin

@configclass
class MeshRandomHoleGridTerrainCfg(SubTerrainBaseCfg):
    """바둑판 랜덤 구멍 지형 설정"""
    
    function = random_hole_grid_terrain
    
    cell_width: float = MISSING
    """각 바둑판 셀의 크기 (m)"""
    
    hole_prob_range: tuple[float, float] = MISSING
    """구멍 확률 범위 (0.0~1.0)"""
    
    platform_size: float = 2.0
    """중앙 플랫폼 크기 (m). 기본값 2.0m × 2.0m"""
    
    min_platform_cells: int = 2
    """최소 플랫폼 셀 개수. platform_size가 너무 작을 때 보장"""

# --------------

POTHOLE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat_terrain": MeshRandomHoleGridTerrainCfg(
            proportion=0.2, cell_width=0.45, hole_prob_range=(0.0, 0.0), platform_size=2.0, min_platform_cells=2
        ),
        "easy": MeshRandomHoleGridTerrainCfg(
            proportion=0.2, cell_width=0.45, hole_prob_range=(0.0, 0.15), platform_size=2.0, min_platform_cells=2
        ),
        "medium": MeshRandomHoleGridTerrainCfg(
            proportion=0.2, cell_width=0.45, hole_prob_range=(0.05, 0.2), platform_size=2.0, min_platform_cells=2
        ),
        "hard": MeshRandomHoleGridTerrainCfg(
            proportion=0.2, cell_width=0.45, hole_prob_range=(0.1, 0.3), platform_size=2.0, min_platform_cells=2
        ),
    },
)
"""Rough terrains configuration."""
