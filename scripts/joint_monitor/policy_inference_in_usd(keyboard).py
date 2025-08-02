# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates policy inference in a prebuilt USD environment.

In this example, we use a locomotion policy to control the H1 robot. The robot was trained
using Isaac-Velocity-Rough-H1-v0. The robot is commanded to move forward at a constant velocity.

.. code-block:: bash

    # Run the script
    ./isaaclab.sh -p scripts/tutorials/03_envs/policy_inference_in_usd.py --checkpoint /path/to/jit/checkpoint.pt

"""

"""Launch Isaac Sim Simulator first."""


import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on inferencing a policy on an H1 robot in a warehouse.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.")#, required=True)
parser.add_argument("--monitor_mode", type=str, default="progress", choices=["progress", "stats"], help="Monitoring mode")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import io
import os
import torch

import omni

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from fuzzy_lab.tasks.locomotion.velocity.config.go2.flat_env_cfg import Go2FlatEnvCfg_PLAY
from isaaclab.devices import Se2Keyboard

from joint_monitoring import AdvancedJointMonitor


def main():
    """Main function."""
    # load the trained jit policy
    if args_cli.checkpoint is None:
        args_cli.checkpoint = "policy.pt"
    policy_path = os.path.abspath(args_cli.checkpoint)
    file_content = omni.client.read_file(policy_path)[2]
    file = io.BytesIO(memoryview(file_content).tobytes())
    policy = torch.jit.load(file, map_location=args_cli.device)

    monitor = AdvancedJointMonitor(monitor_mode="progress")

    # setup environment
    env_cfg = Go2FlatEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.curriculum = None
    env_cfg.scene.terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Environments/Simple_Warehouse/warehouse.usd",
    )
    env_cfg.sim.device = args_cli.device
    if args_cli.device == "cpu":
        env_cfg.sim.use_fabric = False

    # create environment
    env = ManagerBasedRLEnv(cfg=env_cfg)

    # run inference with the policy
    obs, _ = env.reset()
    
    keyboard = Se2Keyboard()
    
    with torch.inference_mode():
        while simulation_app.is_running():
            command = keyboard.advance()
            obs["policy"][:, 9:12] = torch.tensor(command, device=env.device).unsqueeze(0)
            action = policy(obs["policy"])

            # 관절 위치 모니터링
            joint_positions = action[0].cpu().numpy()
            monitor.display_joints(joint_positions)

            obs, _, _, _, _ = env.step(action)


if __name__ == "__main__":
    main()
    simulation_app.close()
