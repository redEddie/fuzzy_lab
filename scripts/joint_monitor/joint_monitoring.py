import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import threading
import queue
import numpy as np
import time

class AdvancedJointMonitor:
    def __init__(self, monitor_mode="progress", history_length=200):
        self.monitor_mode = monitor_mode
        self.history_length = history_length
        
        # 관절 제한값 정의
        self.HIP_LIMIT = np.array([np.deg2rad(-48), np.deg2rad(48)])
        self.HIND_THIGH_LIMIT = np.array([np.deg2rad(-30), np.deg2rad(260)])
        self.FORE_THIGH_LIMIT = np.array([np.deg2rad(-90), np.deg2rad(200)])
        self.CALF_LIMIT = np.array([np.deg2rad(-156), np.deg2rad(-48)])
        
        # 관절 이름과 제한값 매핑 (FL, FR, RL, RR 순서)
        self.joint_names = [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf", 
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf"
        ]
        
        self.joint_limits = [
            self.HIP_LIMIT, self.FORE_THIGH_LIMIT, self.CALF_LIMIT,  # FL
            self.HIP_LIMIT, self.FORE_THIGH_LIMIT, self.CALF_LIMIT,  # FR
            self.HIP_LIMIT, self.HIND_THIGH_LIMIT, self.CALF_LIMIT,  # RL
            self.HIP_LIMIT, self.HIND_THIGH_LIMIT, self.CALF_LIMIT   # RR
        ]
        
        # Go2 default joint positions (FL, FR, RL, RR 순서)
        # From UNITREE_GO2_CFG in isaaclab_assets/robots/unitree.py
        self.default_joint_pos = np.array([
            0.1, 0.8, -1.5,   # FL: hip=0.1, thigh=0.8, calf=-1.5
            -0.1, 0.8, -1.5,  # FR: hip=-0.1, thigh=0.8, calf=-1.5
            0.1, 1.0, -1.5,   # RL: hip=0.1, thigh=1.0, calf=-1.5
            -0.1, 1.0, -1.5   # RR: hip=-0.1, thigh=1.0, calf=-1.5
        ])
        
        # 통계 데이터
        self.joint_history = [deque(maxlen=history_length) for _ in range(12)]
        self.warning_count = [0] * 12
        self.last_update_time = time.time()
        
    def create_enhanced_progress_bar(self, value, min_val, max_val, width=40, name=""):
        """향상된 Progress bar 생성"""
        # 정규화된 값 (0~1)
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))
        
        # Progress bar 세부 문자
        filled = int(normalized * width)
        partial = (normalized * width) % 1
        
        # 부분 채움 문자
        if partial > 0.75:
            partial_char = "▉"
        elif partial > 0.5:
            partial_char = "▌"
        elif partial > 0.25:
            partial_char = "▎"
        else:
            partial_char = ""
            
        bar_filled = "█" * filled
        bar_empty = "░" * (width - filled - (1 if partial_char else 0))
        bar = bar_filled + partial_char + bar_empty
        
        # 상태별 색상
        if normalized > 0.9 or normalized < 0.1:
            color = "\033[91m"  # 빨간색 (위험)
            status = "⚠️ "
        elif normalized > 0.8 or normalized < 0.2:
            color = "\033[93m"  # 노란색 (주의)
            status = "⚡ "
        else:
            color = "\033[92m"  # 초록색 (정상)
            status = "✅ "
        
        reset = "\033[0m"
        
        # 각도 정보 (라디안)
        rad_value = value
        rad_min = min_val
        rad_max = max_val
        
        # 속도 정보 (이전 값과의 차이)
        velocity_info = ""
        if len(self.joint_history[self.joint_names.index(name)]) > 1:
            prev_value = list(self.joint_history[self.joint_names.index(name)])[-1]
            velocity = rad_value - prev_value
            if abs(velocity) > 0.017:  # 0.017 라디안(약 1도) 이상 변화시 표시
                velocity_info = f" (Δ{velocity:+.3f}rad/step)"
        
        return f"{status}{name:10} |{color}{bar}{reset}| {rad_value:7.3f}rad [{rad_min:6.3f}~{rad_max:6.3f}rad]{velocity_info}"
    
    def display_statistics(self):
        """통계 정보 표시 (라디안)"""
        print("\n📊 Joint Statistics:")
        print("-" * 60)
        
        for i, name in enumerate(self.joint_names):
            if len(self.joint_history[i]) > 0:
                positions = list(self.joint_history[i])
                avg_pos = np.mean(positions)
                std_pos = np.std(positions)
                min_pos = np.min(positions)
                max_pos = np.max(positions)
                
                print(f"{name:10}: Avg={avg_pos:6.3f}rad Std={std_pos:5.3f}rad "
                      f"Range=[{min_pos:6.3f}rad, {max_pos:6.3f}rad] Warnings={self.warning_count[i]}")
    
    def display_joints(self, joint_positions):
        """관절 정보 표시 (메인 함수)"""
        current_time = time.time()
        
        # 업데이트 주기 제한
        if current_time - self.last_update_time < 0.05:  # 20Hz
            return
            
        self.last_update_time = current_time
        
        # 정책 액션에 default joint position 추가하여 최종 joint position 계산
        final_joint_positions = joint_positions + self.default_joint_pos
        
        # 히스토리 업데이트 (최종 joint position 사용, 라디안 그대로 저장)
        for i, position in enumerate(final_joint_positions):
            self.joint_history[i].append(position)
        
        if self.monitor_mode == "progress":
            # 화면 클리어
            print("\033[H\033[J", end="")
            
            print("🤖 Advanced Robot Joint Monitor")
            print("=" * 80)
            
            # 다리별로 그룹화하여 표시 (FL, FR, RL, RR 순서)
            leg_names = ["Front Left (FL)", "Front Right (FR)", "Rear Left (RL)", "Rear Right (RR)"]
            
            for leg_idx, leg_name in enumerate(leg_names):
                print(f"\n🦵 {leg_name}:")
                for joint_idx in range(3):
                    i = leg_idx * 3 + joint_idx
                    progress_bar = self.create_enhanced_progress_bar(
                        final_joint_positions[i], 
                        self.joint_limits[i][0], 
                        self.joint_limits[i][1], 
                        width=30, 
                        name=self.joint_names[i]
                    )
                    print(f"  {progress_bar}")
            
            # 요약 통계 (간단히) - 최종 joint position 사용
            warnings = sum(1 for i, pos in enumerate(final_joint_positions) 
                          if pos <= self.joint_limits[i][0] * 1.0 or pos >= self.joint_limits[i][1] * 1.0)
            if warnings > 0:
                print(f"\n⚠️  Active warnings: {warnings} joints")
            else:
                print(f"\n✅ All joints within safe range")
                
            # 추가 정보: 원본 action 및 default position 표시
            print("\n" + "=" * 80)
            print("📊 Raw Data Analysis")
            print("-" * 80)
            
            # 원본 action 값 표시
            print("\n🎯 Policy Action (Raw Output):")
            for leg_idx, leg_name in enumerate(leg_names):
                action_values = []
                for joint_idx in range(3):
                    i = leg_idx * 3 + joint_idx
                    action_values.append(f"{self.joint_names[i]}: {joint_positions[i]:+7.3f}rad")
                print(f"  {leg_name:18} | {' | '.join(action_values)}")
            
            # Default position 값 표시
            print("\n🏠 Default Joint Positions:")
            for leg_idx, leg_name in enumerate(leg_names):
                default_values = []
                for joint_idx in range(3):
                    i = leg_idx * 3 + joint_idx
                    default_values.append(f"{self.joint_names[i]}: {self.default_joint_pos[i]:+7.3f}rad")
                print(f"  {leg_name:18} | {' | '.join(default_values)}")
            
            # 최종 결과 값 표시
            print("\n⚙️ Final Joint Positions (Action + Default):")
            for leg_idx, leg_name in enumerate(leg_names):
                final_values = []
                for joint_idx in range(3):
                    i = leg_idx * 3 + joint_idx
                    final_values.append(f"{self.joint_names[i]}: {final_joint_positions[i]:+7.3f}rad")
                print(f"  {leg_name:18} | {' | '.join(final_values)}")
                
        elif self.monitor_mode == "stats":
            # 통계 모드
            if len(self.joint_history[0]) % 50 == 0:  # 50스텝마다 통계 표시
                print("\033[H\033[J", end="")
                self.display_statistics()