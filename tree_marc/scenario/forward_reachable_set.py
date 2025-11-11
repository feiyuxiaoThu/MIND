"""
前向可达集实现

实现MARC论文中的前向可达集，用于安全保证。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class ForwardReachableSet:
    """前向可达集"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # FRS参数
        self.time_horizon = config.get('time_horizon', 5.0)
        self.safety_margin = config.get('safety_margin', 1.0)
        self.grid_resolution = config.get('grid_resolution', 0.5)
        
    def compute_reachable_set(self, initial_state: np.ndarray, 
                             control_input: np.ndarray,
                             horizon: float) -> np.ndarray:
        """
        计算前向可达集
        
        Args:
            initial_state: 初始状态 [x, y, v, theta, a, delta]
            control_input: 控制输入 [da, ddelta]
            horizon: 时间范围
            
        Returns:
            reachable_set: 可达集边界点
        """
        dt = 0.1
        num_steps = int(horizon / dt)
        
        # 模拟轨迹
        trajectory = self._simulate_trajectory(initial_state, control_input, num_steps)
        
        # 计算可达集边界
        reachable_set = self._compute_reachable_boundary(trajectory)
        
        return reachable_set
        
    def _simulate_trajectory(self, initial_state: np.ndarray, 
                            control_input: np.ndarray,
                            num_steps: int) -> np.ndarray:
        """模拟轨迹"""
        trajectory = np.zeros((num_steps + 1, 6))
        trajectory[0] = initial_state
        
        state = initial_state.copy()
        
        for t in range(num_steps):
            # 自行车模型
            x, y, v, theta, a, delta = state
            da, ddelta = control_input
            
            # 更新状态
            a_next = np.clip(a + da * 0.1, -3.0, 3.0)
            delta_next = np.clip(delta + ddelta * 0.1, -0.5, 0.5)
            
            v_next = v + a_next * 0.1
            v_next = max(0.0, v_next)
            
            x_next = x + v_next * np.cos(theta) * 0.1
            y_next = y + v_next * np.sin(theta) * 0.1
            theta_next = theta + v_next / 2.5 * np.tan(delta_next) * 0.1
            
            trajectory[t + 1] = [x_next, y_next, v_next, theta_next, a_next, delta_next]
            state = trajectory[t + 1]
            
        return trajectory
        
    def _compute_reachable_boundary(self, trajectory: np.ndarray) -> np.ndarray:
        """计算可达集边界"""
        positions = trajectory[:, :2]
        
        # 简化版本：使用凸包
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            boundary_points = positions[hull.vertices]
            
            # 添加安全边界
            center = np.mean(boundary_points, axis=0)
            expanded_points = []
            
            for point in boundary_points:
                direction = point - center
                direction = direction / np.linalg.norm(direction)
                expanded_point = point + direction * self.safety_margin
                expanded_points.append(expanded_point)
                
            return np.array(expanded_points)
            
        except ImportError:
            # 如果没有scipy，使用简单的边界框
            min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
            min_y, max_y = np.min(positions[:, 1]), np.max(positions[:, 1])
            
            boundary_points = np.array([
                [min_x - self.safety_margin, min_y - self.safety_margin],
                [max_x + self.safety_margin, min_y - self.safety_margin],
                [max_x + self.safety_margin, max_y + self.safety_margin],
                [min_x - self.safety_margin, max_y + self.safety_margin]
            ])
            
            return boundary_points
            
    def check_safety(self, trajectory: np.ndarray, 
                    reachable_set: np.ndarray) -> bool:
        """
        检查轨迹安全性
        
        Args:
            trajectory: 轨迹
            reachable_set: 可达集
            
        Returns:
            is_safe: 是否安全
        """
        positions = trajectory[:, :2]
        
        for pos in positions:
            # 检查点是否在可达集内
            if not self._point_in_polygon(pos, reachable_set):
                return False
                
        return True
        
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """检查点是否在多边形内"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
        
    def compute_safety_margin(self, trajectory: np.ndarray, 
                            obstacles: List[np.ndarray]) -> float:
        """
        计算安全边界
        
        Args:
            trajectory: 轨迹
            obstacles: 障碍物列表
            
        Returns:
            min_distance: 最小距离
        """
        min_distance = float('inf')
        
        for pos in trajectory[:, :2]:
            for obstacle in obstacles:
                distance = np.linalg.norm(pos - obstacle)
                min_distance = min(min_distance, distance)
                
        return min_distance
        
    def generate_safe_corridor(self, initial_state: np.ndarray,
                              target_state: np.ndarray,
                              obstacles: List[np.ndarray]) -> Dict[str, Any]:
        """
        生成安全走廊
        
        Args:
            initial_state: 初始状态
            target_state: 目标状态
            obstacles: 障碍物列表
            
        Returns:
            corridor: 安全走廊信息
        """
        # 简化版本：直线走廊
        start_pos = initial_state[:2]
        end_pos = target_state[:2]
        
        # 计算走廊方向
        direction = end_pos - start_pos
        direction = direction / np.linalg.norm(direction)
        
        # 计算垂直方向
        perpendicular = np.array([-direction[1], direction[0]])
        
        # 走廊宽度
        corridor_width = 4.0  # 默认宽度
        
        # 生成走廊边界
        corridor_points = []
        
        for t in np.linspace(0, 1, 10):
            center_point = start_pos + t * (end_pos - start_pos)
            
            # 左边界
            left_point = center_point + perpendicular * corridor_width / 2
            corridor_points.append(left_point)
            
        for t in np.linspace(1, 0, 10):
            center_point = start_pos + t * (end_pos - start_pos)
            
            # 右边界
            right_point = center_point - perpendicular * corridor_width / 2
            corridor_points.append(right_point)
            
        return {
            'boundary_points': np.array(corridor_points),
            'width': corridor_width,
            'center_line': np.array([start_pos, end_pos])
        }