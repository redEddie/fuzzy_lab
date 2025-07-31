# main.py - Go2 로봇 Kinematics 시각화 (모듈화된 버전)
"""Launch Isaac Sim Simulator first."""

import argparse
from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Go2 robot kinematics visualization with modular design.")
parser.add_argument("--checkpoint", type=str, help="Path to model checkpoint exported as jit.")
parser.add_argument("--test-mode", action="store_true", help="Run accuracy test on startup")
parser.add_argument("--legs", nargs="+", choices=["FL", "FR", "RL", "RR"], 
                   help="Select specific legs to visualize (default: all)")

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
from isaaclab.utils.configclass import configclass

# 우리가 만든 모듈들 import
from footstep_kinematics import (
    test_single_leg_accuracy,
    get_foothold_positions,
    calculate_all_legs_kinematics
)
from footstep_visualization import (
    Go2Visualizer,
    debug_robot_info
)


@configclass
class Go2FlatEnvCfg_EVAL(Go2FlatEnvCfg_PLAY):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        
        # remove timeout termination to prevent periodic resetting
        self.terminations.time_out = None


def load_policy(checkpoint_path, device):
    """Policy 파일 로드 (옵션)"""
    
    if not os.path.exists(checkpoint_path):
        print(f"⚠️  Policy file not found: {checkpoint_path}")
        print("   Continuing without policy - robot will use default actions")
        return None
    
    try:
        file_content = omni.client.read_file(checkpoint_path)[2]
        file = io.BytesIO(memoryview(file_content).tobytes())
        policy = torch.jit.load(file, map_location=device)
        print(f"✅ Policy loaded from {checkpoint_path}")
        return policy
    except Exception as e:
        print(f"⚠️  Policy loading failed: {e}")
        print("   Continuing without policy - robot will use default actions")
        return None


def run_accuracy_test(env):
    """정확도 테스트 실행"""
    
    print(f"\n{'='*80}")
    print(f"🔬 KINEMATICS ACCURACY TEST")
    print(f"{'='*80}")
    
    legs_to_test = args_cli.legs if args_cli.legs else ["FL", "FR", "RL", "RR"]
    
    total_avg_error = 0.0
    test_count = 0
    
    for leg in legs_to_test:
        result = test_single_leg_accuracy(env, leg, verbose=True)
        total_avg_error += result['avg_error']
        test_count += 1
    
    overall_avg_error = total_avg_error / test_count if test_count > 0 else 0.0
    
    print(f"\n{'='*80}")
    print(f"📊 OVERALL TEST RESULTS")
    print(f"{'='*80}")
    print(f"Tested legs: {legs_to_test}")
    print(f"Overall average error: {overall_avg_error:.6f}m")
    
    if overall_avg_error < 0.001:
        print("🎉 EXCELLENT! Kinematics implementation is very accurate!")
    elif overall_avg_error < 0.01:
        print("✅ GOOD! Kinematics implementation is reasonably accurate")
    elif overall_avg_error < 0.1:
        print("⚠️  NEEDS IMPROVEMENT: Significant errors detected")
    else:
        print("❌ MAJOR ISSUES: Large errors - kinematics needs major fixes")
    
    print(f"{'='*80}\n")


def main():
    """메인 함수 - 모듈화된 깔끔한 버전"""
    
    print(f"🚀 Go2 Robot Kinematics Visualization")
    print(f"   📍 RED frames: Kinematics calculations")  
    print(f"   📍 BLUE frames: Ground truth from simulation")
    print(f"   🟢 GREEN spheres: Foothold positions")
    print(f"   🎮 Use WASD keys to control the robot")
    
    # 1) Policy 로드 (옵션)
    policy_path = os.path.abspath(args_cli.checkpoint or "policy.pt")
    policy = load_policy(policy_path, args_cli.device)

    # 2) 키보드 제어
    keyboard = Se2Keyboard(
        v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=3.0
    )
    keyboard.reset()

    # 3) 시각화 도구 생성
    visualizer = Go2Visualizer(
        kinematics_scale=0.2,
        groundtruth_scale=0.2,
        foothold_scale=0.1
    )
    
    # 시각화 옵션 설정
    visualizer.set_display_options(
        joint_frames=True,
        footholds=True,
        errors=False,
        legs=args_cli.legs  # 명령행에서 지정한 다리만 표시
    )

    # 4) 환경 설정
    env_cfg = Go2FlatEnvCfg_EVAL()
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

    env = ManagerBasedRLEnv(cfg=env_cfg)

    # 5) 환경 초기화
    obs, _ = env.reset()
    
    # 6) 정확도 테스트 (옵션)
    if args_cli.test_mode:
        run_accuracy_test(env)

    step_count = 0
    print_interval = 30  # 매 30스텝마다 정보 출력
    
    with torch.inference_mode():
        # 메인 시뮬레이션 루프
        while simulation_app.is_running():
            step_count += 1
            
            # a) 키보드 명령
            command = keyboard.advance()
            obs["policy"][:, 9:12] = torch.tensor(command, device=env.device).unsqueeze(0)

            # b) Policy 실행 또는 기본 자세 유지
            if policy is not None:
                action = policy(obs["policy"])
            else:
                # Policy가 없으면 기본 관절 위치 유지
                action = env.scene["robot"].data.default_joint_pos.clone()
            
            obs, _, _, _, _ = env.step(action)

            # c) 시각화 업데이트
            print_info = (step_count % print_interval == 0)
            print_errors = print_info  # 정보 출력할 때만 오차도 출력
            
            success = visualizer.update(env, print_info=print_info, print_errors=print_errors)
            
            if print_info:
                print(f"📊 Step {step_count} - Visualization {'successful' if success else 'failed'}")
                
                # 발끝 위치 정보 (간단히)
                foothold_info, base_pos, base_quat = get_foothold_positions(env)
                print(f"   🏃 Robot base: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
                
                # 선택된 다리만 표시
                legs_to_show = args_cli.legs if args_cli.legs else ["FL", "FR", "RL", "RR"]
                for leg in legs_to_show:
                    if leg in foothold_info:
                        pos = foothold_info[leg]['world_position']
                        print(f"   🦶 {leg} foot: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("🔚 Closing simulation...")
        simulation_app.close()