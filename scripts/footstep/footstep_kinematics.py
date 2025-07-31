# go2_kinematics.py
"""Go2 로봇의 Forward Kinematics 모듈"""

import torch
import math
from isaaclab.utils.math import quat_apply


def get_robot_base_pose(env):
    """로봇 base의 위치와 방향을 가져오기"""
    robot = env.scene["robot"]
    body_positions = robot.data.body_pos_w[0]
    body_orientations = robot.data.body_quat_w[0]  # [w, x, y, z]
    body_names = robot.body_names
    
    base_idx = body_names.index("base")
    base_pos = body_positions[base_idx, :3]
    base_quat = body_orientations[base_idx]  # [w, x, y, z]
    
    return base_pos, base_quat


def transform_to_world_frame(base_relative_pos, base_pos, base_quat):
    """Base coordinate에서 World coordinate로 변환"""
    
    # Quaternion을 사용해서 base frame의 상대 위치를 world frame으로 회전
    rotated_pos = quat_apply(base_quat.unsqueeze(0), base_relative_pos.unsqueeze(0))[0]
    
    # World frame 위치 = base 위치 + 회전된 상대 위치
    world_pos = base_pos + rotated_pos
    
    return world_pos


def get_joint_angles_for_leg(env, leg_name):
    """특정 다리의 관절 각도를 추출"""
    robot = env.scene["robot"]
    joint_names = robot.data.joint_names
    joint_pos = robot.data.joint_pos[0]
    
    target_joints = [f"{leg_name}_hip_joint", f"{leg_name}_thigh_joint", f"{leg_name}_calf_joint"]
    joint_angles = []
    
    for target_joint in target_joints:
        if target_joint in joint_names:
            joint_idx = joint_names.index(target_joint)
            angle = joint_pos[joint_idx].item()
            joint_angles.append(angle)
        else:
            print(f"Warning: Joint {target_joint} not found")
            joint_angles.append(0.0)
    
    return joint_angles


def calculate_go2_forward_kinematics(joint_angles, leg_name, device='cuda:0', verbose=False):
    """
    Go2 로봇의 Forward Kinematics (Base frame 기준)
    
    Args:
        joint_angles: [hip_angle, thigh_angle, calf_angle] in radians
        leg_name: "FL", "FR", "RL", "RR"
        device: torch device
        verbose: 상세 출력 여부
        
    Returns:
        dict: {'hip': pos, 'thigh': pos, 'calf': pos, 'foot': pos} - base frame 기준
    """
    
    hip_angle, thigh_angle, calf_angle = joint_angles
    
    # Go2 로봇의 정확한 링크 길이 및 오프셋
    HIP_OFFSETS = {
        'FL': [0.1934, 0.0465, 0.0],      # Front Left
        'FR': [0.1934, -0.0465, 0.0],     # Front Right  
        'RL': [-0.1934, 0.0465, 0.0],     # Rear Left
        'RR': [-0.1934, -0.0465, 0.0]     # Rear Right
    }
    
    HIP_TO_THIGH_OFFSET = 0.08505  # Y 방향
    THIGH_LENGTH = 0.213           # Z 방향 (downward)
    CALF_LENGTH = 0.213            # Z 방향 (downward)
    
    if verbose:
        print(f"\n--- {leg_name} Forward Kinematics ---")
        print(f"Joint angles: hip={math.degrees(hip_angle):.1f}°, "
              f"thigh={math.degrees(thigh_angle):.1f}°, "
              f"calf={math.degrees(calf_angle):.1f}°")
    
    # 1. Hip joint 위치 (base frame 기준)
    hip_pos = torch.tensor(HIP_OFFSETS[leg_name], dtype=torch.float32, device=device)
    if verbose:
        print(f"Hip position: {hip_pos.tolist()}")
    
    # 2. Hip에서 Thigh로의 변환 (Hip motor: X축 회전)
    y_sign = 1.0 if leg_name.endswith('L') else -1.0
    hip_to_thigh_local = torch.tensor([0.0, y_sign * HIP_TO_THIGH_OFFSET, 0.0], 
                                     dtype=torch.float32, device=device)
    
    # X축 회전 적용
    hip_rotated_y = hip_to_thigh_local[1] * math.cos(hip_angle) - hip_to_thigh_local[2] * math.sin(hip_angle)
    hip_rotated_z = hip_to_thigh_local[1] * math.sin(hip_angle) + hip_to_thigh_local[2] * math.cos(hip_angle)
    
    thigh_pos = hip_pos + torch.tensor([0.0, hip_rotated_y, hip_rotated_z], dtype=torch.float32, device=device)
    if verbose:
        print(f"Thigh position: {thigh_pos.tolist()}")
    
    # 3. Thigh에서 Calf로의 변환 (Thigh motor: Y축 회전)
    thigh_to_calf_local = torch.tensor([0.0, 0.0, -THIGH_LENGTH], dtype=torch.float32, device=device)
    
    # Y축 회전 적용
    thigh_rotated_x = thigh_to_calf_local[0] * math.cos(thigh_angle) + thigh_to_calf_local[2] * math.sin(thigh_angle)
    thigh_rotated_z = -thigh_to_calf_local[0] * math.sin(thigh_angle) + thigh_to_calf_local[2] * math.cos(thigh_angle)
    
    # Hip 회전의 영향 적용
    thigh_to_calf_rotated = torch.tensor([thigh_rotated_x, 0.0, thigh_rotated_z], dtype=torch.float32, device=device)
    
    # Hip X축 회전 적용
    final_y = thigh_to_calf_rotated[1] * math.cos(hip_angle) - thigh_to_calf_rotated[2] * math.sin(hip_angle)
    final_z = thigh_to_calf_rotated[1] * math.sin(hip_angle) + thigh_to_calf_rotated[2] * math.cos(hip_angle)
    
    calf_pos = thigh_pos + torch.tensor([thigh_to_calf_rotated[0], final_y, final_z], dtype=torch.float32, device=device)
    if verbose:
        print(f"Calf position: {calf_pos.tolist()}")
    
    # 4. Calf에서 Foot로의 변환 (Calf motor: Y축 회전)
    calf_to_foot_local = torch.tensor([0.0, 0.0, -CALF_LENGTH], dtype=torch.float32, device=device)
    
    # Y축 회전 적용 (Calf)
    calf_rotated_x = calf_to_foot_local[0] * math.cos(calf_angle) + calf_to_foot_local[2] * math.sin(calf_angle)
    calf_rotated_z = -calf_to_foot_local[0] * math.sin(calf_angle) + calf_to_foot_local[2] * math.cos(calf_angle)
    
    # Y축 회전 적용 (Thigh의 영향)
    temp_x = calf_rotated_x * math.cos(thigh_angle) + calf_rotated_z * math.sin(thigh_angle)  
    temp_z = -calf_rotated_x * math.sin(thigh_angle) + calf_rotated_z * math.cos(thigh_angle)
    
    calf_to_foot_final = torch.tensor([temp_x, 0.0, temp_z], dtype=torch.float32, device=device)
    
    # X축 회전 적용 (Hip의 영향)
    final_foot_y = calf_to_foot_final[1] * math.cos(hip_angle) - calf_to_foot_final[2] * math.sin(hip_angle)
    final_foot_z = calf_to_foot_final[1] * math.sin(hip_angle) + calf_to_foot_final[2] * math.cos(hip_angle)
    
    foot_pos = calf_pos + torch.tensor([calf_to_foot_final[0], final_foot_y, final_foot_z], dtype=torch.float32, device=device)
    if verbose:
        print(f"Foot position: {foot_pos.tolist()}")
    
    return {
        'hip': hip_pos,
        'thigh': thigh_pos,
        'calf': calf_pos,
        'foot': foot_pos
    }


def calculate_all_legs_kinematics(env, verbose=False):
    """모든 다리의 kinematics를 계산하고 world frame으로 변환"""
    
    legs = ["FL", "FR", "RL", "RR"]
    joint_types = ["hip", "thigh", "calf", "foot"]
    
    # 로봇 base pose
    base_pos, base_quat = get_robot_base_pose(env)
    
    all_results = {}
    
    for leg in legs:
        # 관절 각도 가져오기
        angles = get_joint_angles_for_leg(env, leg)
        
        # Base frame에서 kinematics 계산
        base_positions = calculate_go2_forward_kinematics(angles, leg, device=env.device, verbose=verbose)
        
        # World frame으로 변환
        world_positions = {}
        for joint_type, base_pos_rel in base_positions.items():
            world_pos = transform_to_world_frame(base_pos_rel, base_pos, base_quat)
            world_positions[joint_type] = world_pos
        
        all_results[leg] = {
            'joint_angles': angles,
            'base_positions': base_positions,
            'world_positions': world_positions
        }
    
    return all_results, base_pos, base_quat


def test_single_leg_accuracy(env, leg_name="FL", verbose=True):
    """단일 다리의 정확도 테스트"""
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"ACCURACY TEST - {leg_name} LEG")
        print(f"{'='*80}")
    
    robot = env.scene["robot"]
    
    # 로봇 base pose
    base_pos, base_quat = get_robot_base_pose(env)
    
    if verbose:
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
        
        print(f"Robot base: pos={base_pos.tolist()}, "
              f"euler=[{math.degrees(roll):.1f}°, {math.degrees(pitch):.1f}°, {math.degrees(yaw):.1f}°]")
    
    # 관절 각도
    angles = get_joint_angles_for_leg(env, leg_name)
    if verbose:
        print(f"Joint angles: {[math.degrees(a) for a in angles]} degrees")
    
    # Ground Truth 위치들
    body_positions = robot.data.body_pos_w[0]
    body_names = robot.body_names
    
    joint_types = ["hip", "thigh", "calf", "foot"]
    gt_world_positions = {}
    
    if verbose:
        print(f"\n--- GROUND TRUTH (World Frame) ---")
    
    for joint_type in joint_types:
        joint_body_name = f"{leg_name}_{joint_type}"
        if joint_body_name in body_names:
            body_idx = body_names.index(joint_body_name)
            gt_world_pos = body_positions[body_idx, :3]
            gt_world_positions[joint_type] = gt_world_pos
            if verbose:
                print(f"GT {joint_type:5s}: {gt_world_pos.tolist()}")
    
    # Kinematics 계산
    base_positions = calculate_go2_forward_kinematics(angles, leg_name, device=env.device, verbose=verbose)
    
    # World frame 변환
    kin_world_positions = {}
    if verbose:
        print(f"\n--- KINEMATICS RESULTS (World Frame) ---")
    
    for joint_type, base_rel_pos in base_positions.items():
        world_pos = transform_to_world_frame(base_rel_pos, base_pos, base_quat)
        kin_world_positions[joint_type] = world_pos
        if verbose:
            print(f"KIN {joint_type:5s}: {world_pos.tolist()}")
    
    # 오차 분석
    if verbose:
        print(f"\n--- ERROR ANALYSIS ---")
    
    errors = {}
    total_error = 0.0
    error_count = 0
    
    for joint_type in joint_types:
        if joint_type in gt_world_positions and joint_type in kin_world_positions:
            gt_pos = gt_world_positions[joint_type]
            kin_pos = kin_world_positions[joint_type]
            error = torch.norm(gt_pos - kin_pos).item()
            
            errors[joint_type] = error
            total_error += error
            error_count += 1
            
            if verbose:
                print(f"{joint_type:5s} error: {error:.6f}m")
                print(f"       DIFF: {(gt_pos - kin_pos).tolist()}")
    
    avg_error = total_error / error_count if error_count > 0 else 0.0
    
    if verbose:
        print(f"\nAverage error: {avg_error:.6f}m")
        
        if avg_error < 0.001:
            print("🎉 EXCELLENT! Kinematics is very accurate!")
        elif avg_error < 0.01:
            print("✅ GOOD! Kinematics is reasonably accurate")
        elif avg_error < 0.1:
            print("⚠️  NEEDS IMPROVEMENT: Significant errors detected")
        else:
            print("❌ MAJOR ISSUES: Large errors - kinematics needs major fixes")
    
    return {
        'leg_name': leg_name,
        'joint_angles': angles,
        'errors': errors,
        'avg_error': avg_error,
        'gt_positions': gt_world_positions,
        'kin_positions': kin_world_positions
    }


def get_foothold_positions(env):
    """모든 발의 위치를 가져오기 (Kinematics 기반)"""
    
    legs = ["FL", "FR", "RL", "RR"]
    foothold_positions = {}
    
    # 로봇 base pose
    base_pos, base_quat = get_robot_base_pose(env)
    
    for leg in legs:
        # 관절 각도 가져오기
        angles = get_joint_angles_for_leg(env, leg)
        
        # Base frame에서 발 위치 계산
        base_positions = calculate_go2_forward_kinematics(angles, leg, device=env.device, verbose=False)
        foot_base_pos = base_positions['foot']
        
        # World frame으로 변환
        foot_world_pos = transform_to_world_frame(foot_base_pos, base_pos, base_quat)
        
        foothold_positions[leg] = {
            'world_position': foot_world_pos,
            'base_relative': foot_base_pos,
            'joint_angles': angles
        }
    
    return foothold_positions, base_pos, base_quat