# go2_visualization.py
"""Go2 로봇 시각화 모듈"""

import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

from footstep_kinematics import (
    get_robot_base_pose, 
    get_joint_angles_for_leg, 
    calculate_go2_forward_kinematics,
    transform_to_world_frame
)


def create_joint_visualizers(kinematics_scale=0.4, groundtruth_scale=0.4):
    """관절 좌표계 시각화용 마커들 생성"""
    
    # Kinematics 결과용 visualizer (빨간색)
    kinematics_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/kinematics_frames",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(kinematics_scale, kinematics_scale, kinematics_scale),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0))  # 빨간색
            )
        }
    )
    
    # Ground truth용 visualizer (파란색)
    groundtruth_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/groundtruth_frames",
        markers={
            "frame": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                scale=(groundtruth_scale, groundtruth_scale, groundtruth_scale),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0))  # 파란색
            )
        }
    )
    
    return VisualizationMarkers(kinematics_cfg), VisualizationMarkers(groundtruth_cfg)


def create_foothold_visualizer(scale=0.3):
    """발끝 위치 시각화용 마커 생성"""
    
    foothold_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/foothold_markers",
        markers={
            "sphere": sim_utils.SphereCfg(
                radius=scale,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0))  # 초록색
            )
        }
    )
    
    return VisualizationMarkers(foothold_cfg)


def create_error_visualizer(scale=0.05):
    """오차 벡터 시각화용 마커 생성"""
    
    error_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/error_vectors",
        markers={
            "arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(scale, scale, scale),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0))  # 노란색
            )
        }
    )
    
    return VisualizationMarkers(error_cfg)


def get_groundtruth_data_by_name(env):
    """시뮬레이션의 Ground Truth 데이터를 이름으로 인덱싱"""
    robot = env.scene["robot"]
    body_positions = robot.data.body_pos_w[0]
    body_orientations = robot.data.body_quat_w[0]
    body_names = robot.body_names
    
    gt_data = {}
    
    for i, body_name in enumerate(body_names):
        gt_data[body_name] = {
            'position': body_positions[i, :3],
            'orientation': body_orientations[i],
            'index': i
        }
    
    return gt_data


def get_joint_orientation(joint_type, leg_name, joint_angles, device):
    """관절의 방향을 계산 (향후 개선 가능)"""
    
    # 현재는 기본 방향 반환 (향후 정확한 방향 계산으로 개선 가능)
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)


def visualize_joint_frames(env, kinematics_visualizer, groundtruth_visualizer, 
                          selected_legs=None, print_errors=False):
    """
    관절 좌표계 시각화
    
    Args:
        env: Isaac Lab 환경
        kinematics_visualizer: Kinematics 결과 시각화용
        groundtruth_visualizer: Ground Truth 시각화용
        selected_legs: 시각화할 다리 리스트 (None이면 모든 다리)
        print_errors: 오차 정보 출력 여부
    
    Returns:
        tuple: (success, avg_error, max_error)
    """
    
    legs = selected_legs if selected_legs is not None else ["FL", "FR", "RL", "RR"]
    joint_types = ["hip", "thigh", "calf", "foot"]
    
    # 로봇 base pose
    base_pos, base_quat = get_robot_base_pose(env)
    
    # Ground Truth 데이터
    gt_data = get_groundtruth_data_by_name(env)
    
    kin_positions = []
    kin_orientations = []
    gt_positions = []
    gt_orientations = []
    
    position_errors = []
    
    for leg in legs:
        angles = get_joint_angles_for_leg(env, leg)
        
        # Base frame에서 kinematics 계산
        kin_base_dict = calculate_go2_forward_kinematics(angles, leg, device=env.device, verbose=False)
        
        for joint_type in joint_types:
            joint_body_name = f"{leg}_{joint_type}"
            
            # Kinematics: Base frame → World frame 변환
            base_rel_pos = kin_base_dict[joint_type]
            world_pos = transform_to_world_frame(base_rel_pos, base_pos, base_quat)
            
            # 관절 방향 (향후 개선 가능)
            joint_orientation = get_joint_orientation(joint_type, leg, angles, env.device)
            
            kin_positions.append(world_pos)
            kin_orientations.append(joint_orientation)
            
            # Ground Truth
            if joint_body_name in gt_data:
                gt_pos = gt_data[joint_body_name]['position']
                gt_ori = gt_data[joint_body_name]['orientation']
                
                gt_positions.append(gt_pos)
                gt_orientations.append(gt_ori)
                
                # 오차 계산
                error = torch.norm(world_pos - gt_pos).item()
                position_errors.append(error)
                
            else:
                gt_positions.append(world_pos)
                gt_orientations.append(joint_orientation)
                position_errors.append(0.0)
    
    # 시각화 실행
    if kin_positions and gt_positions:
        if len(kin_positions) == len(gt_positions):
            try:
                # Kinematics 시각화 (빨간색)
                kin_pos_tensor = torch.stack(kin_positions, dim=0)
                kin_ori_tensor = torch.stack(kin_orientations, dim=0)
                kin_scales = torch.ones((kin_pos_tensor.shape[0], 3), device=env.device) * 0.4
                kin_env_indices = torch.zeros(kin_pos_tensor.shape[0], dtype=torch.long, device=env.device)
                
                kinematics_visualizer.visualize(kin_pos_tensor, kin_ori_tensor, kin_scales, kin_env_indices)
                
                # Ground Truth 시각화 (파란색)
                gt_pos_tensor = torch.stack(gt_positions, dim=0)
                gt_ori_tensor = torch.stack(gt_orientations, dim=0)
                gt_scales = torch.ones((gt_pos_tensor.shape[0], 3), device=env.device) * 0.4
                gt_env_indices = torch.zeros(gt_pos_tensor.shape[0], dtype=torch.long, device=env.device)
                
                groundtruth_visualizer.visualize(gt_pos_tensor, gt_ori_tensor, gt_scales, gt_env_indices)
                
                # 오차 통계
                avg_error = sum(position_errors) / len(position_errors) if position_errors else 0.0
                max_error = max(position_errors) if position_errors else 0.0
                
                if print_errors and avg_error > 0:
                    print(f"   📏 Joint frame errors - Avg: {avg_error:.4f}m, Max: {max_error:.4f}m")
                
                return True, avg_error, max_error
                
            except Exception as e:
                if print_errors:
                    print(f"❌ Joint frame visualization failed: {e}")
                return False, 0.0, 0.0
    
    return False, 0.0, 0.0


def visualize_footholds(env, foothold_visualizer, print_info=False):
    """
    발끝 위치 시각화
    
    Args:
        env: Isaac Lab 환경
        foothold_visualizer: 발끝 시각화용 마커
        print_info: 정보 출력 여부
    
    Returns:
        dict: 각 다리의 발끝 위치 정보
    """
    
    legs = ["FL", "FR", "RL", "RR"]
    
    # 로봇 base pose
    base_pos, base_quat = get_robot_base_pose(env)
    
    foothold_positions = []
    foothold_orientations = []
    foothold_info = {}
    
    if print_info:
        print(f"   🦶 Foothold positions:")
    
    for leg in legs:
        # 관절 각도 가져오기
        angles = get_joint_angles_for_leg(env, leg)
        
        # Base frame에서 발 위치 계산
        base_positions = calculate_go2_forward_kinematics(angles, leg, device=env.device, verbose=False)
        foot_base_pos = base_positions['foot']
        
        # World frame으로 변환
        foot_world_pos = transform_to_world_frame(foot_base_pos, base_pos, base_quat)
        
        foothold_positions.append(foot_world_pos)
        foothold_orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device))
        
        foothold_info[leg] = {
            'world_position': foot_world_pos,
            'base_relative': foot_base_pos,
            'joint_angles': angles
        }
        
        if print_info:
            print(f"      {leg}: [{foot_world_pos[0]:.3f}, {foot_world_pos[1]:.3f}, {foot_world_pos[2]:.3f}]")
    
    # 시각화 실행
    if foothold_positions:
        try:
            pos_tensor = torch.stack(foothold_positions, dim=0)
            ori_tensor = torch.stack(foothold_orientations, dim=0)
            scales = torch.ones((pos_tensor.shape[0], 3), device=env.device) * 0.3
            env_indices = torch.zeros(pos_tensor.shape[0], dtype=torch.long, device=env.device)
            
            foothold_visualizer.visualize(pos_tensor, ori_tensor, scales, env_indices)
            
        except Exception as e:
            if print_info:
                print(f"❌ Foothold visualization failed: {e}")
    
    return foothold_info


def visualize_errors(env, error_visualizer, kinematics_positions, groundtruth_positions, 
                    print_details=False):
    """
    오차 벡터 시각화
    
    Args:
        env: Isaac Lab 환경
        error_visualizer: 오차 시각화용 마커
        kinematics_positions: Kinematics 계산 결과 위치들
        groundtruth_positions: Ground Truth 위치들
        print_details: 상세 정보 출력 여부
    
    Returns:
        dict: 오차 통계 정보
    """
    
    if len(kinematics_positions) != len(groundtruth_positions):
        if print_details:
            print(f"Warning: Position count mismatch - KIN: {len(kinematics_positions)}, GT: {len(groundtruth_positions)}")
        return {}
    
    error_positions = []
    error_orientations = []
    error_scales = []
    error_magnitudes = []
    
    for kin_pos, gt_pos in zip(kinematics_positions, groundtruth_positions):
        # 오차 벡터 계산
        error_vector = gt_pos - kin_pos
        error_magnitude = torch.norm(error_vector).item()
        error_magnitudes.append(error_magnitude)
        
        # 오차 벡터의 중점에 화살표 표시
        midpoint = (kin_pos + gt_pos) / 2.0
        error_positions.append(midpoint)
        
        # 기본 방향 (향후 오차 벡터 방향으로 개선 가능)
        error_orientations.append(torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device))
        
        # 오차 크기에 비례한 스케일 (최소 0.01, 최대 0.5)
        scale_factor = min(max(error_magnitude * 10.0, 0.01), 0.5)
        error_scales.append([scale_factor, scale_factor, scale_factor])
    
    # 시각화 실행
    if error_positions:
        try:
            error_pos_tensor = torch.stack(error_positions, dim=0)
            error_ori_tensor = torch.stack(error_orientations, dim=0)
            error_scale_tensor = torch.tensor(error_scales, device=env.device, dtype=torch.float32)
            error_env_indices = torch.zeros(error_pos_tensor.shape[0], dtype=torch.long, device=env.device)
            
            error_visualizer.visualize(error_pos_tensor, error_ori_tensor, error_scale_tensor, error_env_indices)
            
        except Exception as e:
            if print_details:
                print(f"❌ Error visualization failed: {e}")
    
    # 통계 계산
    if error_magnitudes:
        avg_error = sum(error_magnitudes) / len(error_magnitudes)
        max_error = max(error_magnitudes)
        min_error = min(error_magnitudes)
        
        error_stats = {
            'average': avg_error,
            'maximum': max_error,
            'minimum': min_error,
            'count': len(error_magnitudes),
            'individual_errors': error_magnitudes
        }
        
        if print_details:
            print(f"   📊 Error Statistics:")
            print(f"      Average: {avg_error:.6f}m")
            print(f"      Maximum: {max_error:.6f}m")
            print(f"      Minimum: {min_error:.6f}m")
            print(f"      Count: {len(error_magnitudes)} joints")
        
        return error_stats
    
    return {}


def debug_robot_info(env, print_joint_info=True, print_body_info=False):
    """로봇 정보 디버깅 출력"""
    
    robot = env.scene["robot"]
    base_pos, base_quat = get_robot_base_pose(env)
    
    # Base orientation을 Euler angle로 변환
    w, x, y, z = base_quat.tolist()
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x*x + y*y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = math.copysign(math.pi/2, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    
    print(f"\n🤖 Robot State:")
    print(f"   Base position: [{base_pos[0]:.3f}, {base_pos[1]:.3f}, {base_pos[2]:.3f}]")
    print(f"   Base orientation: Roll={math.degrees(roll):.1f}°, Pitch={math.degrees(pitch):.1f}°, Yaw={math.degrees(yaw):.1f}°")
    
    if print_joint_info:
        print(f"   Joint angles:")
        legs = ["FL", "FR", "RL", "RR"]
        for leg in legs:
            angles = get_joint_angles_for_leg(env, leg)
            angles_deg = [math.degrees(a) for a in angles]
            print(f"      {leg}: [{angles_deg[0]:6.1f}°, {angles_deg[1]:6.1f}°, {angles_deg[2]:6.1f}°]")
    
    if print_body_info:
        print(f"\n📋 All body names ({len(robot.body_names)} total):")
        for i, name in enumerate(robot.body_names):
            pos = robot.data.body_pos_w[0, i, :3]
            print(f"   {i:2d}: {name:15s} -> [{pos[0]:7.4f}, {pos[1]:7.4f}, {pos[2]:7.4f}]")


class Go2Visualizer:
    """Go2 로봇 시각화 통합 클래스"""
    
    def __init__(self, kinematics_scale=0.4, groundtruth_scale=0.4, 
                 foothold_scale=0.3, error_scale=0.05):
        """
        Args:
            kinematics_scale: Kinematics 좌표계 크기
            groundtruth_scale: Ground Truth 좌표계 크기
            foothold_scale: 발끝 마커 크기
            error_scale: 오차 벡터 크기
        """
        
        # 시각화 도구들 생성
        self.kinematics_viz, self.groundtruth_viz = create_joint_visualizers(
            kinematics_scale, groundtruth_scale
        )
        self.foothold_viz = create_foothold_visualizer(foothold_scale)
        self.error_viz = create_error_visualizer(error_scale)
        
        # 설정
        self.show_joint_frames = True
        self.show_footholds = True
        self.show_errors = False
        self.selected_legs = None  # None이면 모든 다리
        
        # 통계
        self.last_error_stats = {}
    
    def update(self, env, print_info=False, print_errors=False):
        """모든 시각화 업데이트"""
        
        success_count = 0
        total_avg_error = 0.0
        
        # 1) 관절 좌표계 시각화
        if self.show_joint_frames:
            success, avg_error, max_error = visualize_joint_frames(
                env, self.kinematics_viz, self.groundtruth_viz,
                selected_legs=self.selected_legs, print_errors=print_errors
            )
            
            if success:
                success_count += 1
                total_avg_error += avg_error
                
                if print_info:
                    print(f"   📍 Joint frames: {16 if self.selected_legs is None else len(self.selected_legs)*4} markers")
        
        # 2) 발끝 위치 시각화
        if self.show_footholds:
            foothold_info = visualize_footholds(env, self.foothold_viz, print_info=print_info)
            if foothold_info:
                success_count += 1
        
        # 3) 디버깅 정보 출력
        if print_info:
            debug_robot_info(env, print_joint_info=True, print_body_info=False)
        
        return success_count > 0
    
    def set_display_options(self, joint_frames=True, footholds=True, errors=False, legs=None):
        """시각화 옵션 설정"""
        self.show_joint_frames = joint_frames
        self.show_footholds = footholds  
        self.show_errors = errors
        self.selected_legs = legs
    
    def get_error_stats(self):
        """마지막 오차 통계 반환"""
        return self.last_error_stats