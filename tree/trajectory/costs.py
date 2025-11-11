"""
成本函数和势场模块

实现轨迹优化中的各种成本函数和势场方法。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from ..utils.geometry import GeometryUtils


class CostFunction(ABC):
    """成本函数基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weight = config.get('weight', 1.0)
        
    @abstractmethod
    def compute(self, state: np.ndarray, control: np.ndarray, 
                target: Optional[np.ndarray] = None) -> float:
        """计算成本"""
        pass
        
    @abstractmethod
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        pass


class QuadraticCost(CostFunction):
    """二次成本函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.Q = config.get('Q', np.eye(6))  # 状态权重矩阵
        self.R = config.get('R', np.eye(2))  # 控制权重矩阵
        self.ref_state = config.get('ref_state', np.zeros(6))
        
    def compute(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> float:
        """计算二次成本"""
        if target is not None:
            self.ref_state = target
            
        state_error = state - self.ref_state
        state_cost = state_error.T @ self.Q @ state_error
        control_cost = control.T @ self.R @ control
        
        return self.weight * (state_cost + control_cost)
        
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        if target is not None:
            self.ref_state = target
            
        state_error = state - self.ref_state
        
        # 状态成本梯度
        state_grad = 2 * self.weight * self.Q @ state_error
        
        # 控制成本梯度
        control_grad = 2 * self.weight * self.R @ control
        
        return state_grad, control_grad


class SafetyCost(CostFunction):
    """安全成本函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.safety_distance = config.get('safety_distance', 2.0)
        self.collision_penalty = config.get('collision_penalty', 1000.0)
        self.obstacles = []
        
    def set_obstacles(self, obstacles: List[np.ndarray]):
        """设置障碍物位置"""
        self.obstacles = obstacles
        
    def compute(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> float:
        """计算安全成本"""
        position = state[:2]
        total_cost = 0.0
        
        for obstacle in self.obstacles:
            distance = GeometryUtils.euclidean_distance(position, obstacle)
            
            if distance < self.safety_distance:
                # 使用指数函数增加接近障碍物的成本
                cost = self.collision_penalty * np.exp(-(distance / self.safety_distance))
                total_cost += cost
                
        return self.weight * total_cost
        
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        position = state[:2]
        state_grad = np.zeros_like(state)
        control_grad = np.zeros_like(control)
        
        for obstacle in self.obstacles:
            distance = GeometryUtils.euclidean_distance(position, obstacle)
            
            if distance < self.safety_distance and distance > 0.01:
                # 计算梯度方向
                direction = (position - obstacle) / distance
                
                # 指数函数的导数
                exp_term = np.exp(-(distance / self.safety_distance))
                grad_magnitude = self.weight * self.collision_penalty * exp_term / self.safety_distance
                
                state_grad[:2] += direction * grad_magnitude
                
        return state_grad, control_grad


class TargetCost(CostFunction):
    """目标成本函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_position = config.get('target_position', np.zeros(2))
        self.target_velocity = config.get('target_velocity', 10.0)
        self.position_weight = config.get('position_weight', 1.0)
        self.velocity_weight = config.get('velocity_weight', 0.5)
        
    def set_target(self, position: np.ndarray, velocity: float):
        """设置目标"""
        self.target_position = position
        self.target_velocity = velocity
        
    def compute(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> float:
        """计算目标成本"""
        position = state[:2]
        velocity = state[2]
        
        # 位置偏差成本
        position_error = GeometryUtils.euclidean_distance(position, self.target_position)
        position_cost = self.position_weight * position_error**2
        
        # 速度偏差成本
        velocity_error = velocity - self.target_velocity
        velocity_cost = self.velocity_weight * velocity_error**2
        
        return self.weight * (position_cost + velocity_cost)
        
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        position = state[:2]
        velocity = state[2]
        
        state_grad = np.zeros_like(state)
        control_grad = np.zeros_like(control)
        
        # 位置梯度
        if GeometryUtils.euclidean_distance(position, self.target_position) > 0.01:
            position_direction = (position - self.target_position) / GeometryUtils.euclidean_distance(position, self.target_position)
            state_grad[:2] = 2 * self.weight * self.position_weight * position_direction
            
        # 速度梯度
        state_grad[2] = 2 * self.weight * self.velocity_weight * (velocity - self.target_velocity)
        
        return state_grad, control_grad


class ComfortCost(CostFunction):
    """舒适性成本函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_acceleration = config.get('max_acceleration', 3.0)
        self.max_steering_rate = config.get('max_steering_rate', 0.5)
        self.acceleration_weight = config.get('acceleration_weight', 1.0)
        self.steering_weight = config.get('steering_weight', 1.0)
        
    def compute(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> float:
        """计算舒适性成本"""
        acceleration = control[0]
        steering_rate = control[1]
        
        # 加速度成本（惩罚过大的加速度）
        acc_cost = self.acceleration_weight * (acceleration / self.max_acceleration)**2
        
        # 转向速率成本（惩罚过快的转向）
        steering_cost = self.steering_weight * (steering_rate / self.max_steering_rate)**2
        
        return self.weight * (acc_cost + steering_cost)
        
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算梯度"""
        acceleration = control[0]
        steering_rate = control[1]
        
        state_grad = np.zeros_like(state)
        control_grad = np.zeros_like(control)
        
        # 控制梯度
        control_grad[0] = 2 * self.weight * self.acceleration_weight * acceleration / (self.max_acceleration**2)
        control_grad[1] = 2 * self.weight * self.steering_weight * steering_rate / (self.max_steering_rate**2)
        
        return state_grad, control_grad


class PotentialField:
    """势场类"""
    
    def __init__(self, offsets: np.ndarray, resolution: float, 
                 xx: np.ndarray, yy: np.ndarray, field: np.ndarray):
        self.offsets = offsets
        self.resolution = resolution
        self.xx = xx
        self.yy = yy
        self.field = field
        
    def get_value(self, position: np.ndarray) -> float:
        """获取指定位置的势场值"""
        # 转换到网格坐标
        grid_x = int((position[0] - self.offsets[0]) / self.resolution)
        grid_y = int((position[1] - self.offsets[1]) / self.resolution)
        
        # 边界检查
        grid_x = np.clip(grid_x, 0, self.field.shape[1] - 1)
        grid_y = np.clip(grid_y, 0, self.field.shape[0] - 1)
        
        return self.field[grid_y, grid_x]
        
    def get_gradient(self, position: np.ndarray) -> np.ndarray:
        """获取指定位置的势场梯度"""
        # 转换到网格坐标
        grid_x = int((position[0] - self.offsets[0]) / self.resolution)
        grid_y = int((position[1] - self.offsets[1]) / self.resolution)
        
        # 边界检查
        grid_x = np.clip(grid_x, 1, self.field.shape[1] - 2)
        grid_y = np.clip(grid_y, 1, self.field.shape[0] - 2)
        
        # 数值梯度
        dx = (self.field[grid_y, grid_x + 1] - self.field[grid_y, grid_x - 1]) / (2 * self.resolution)
        dy = (self.field[grid_y + 1, grid_x] - self.field[grid_y - 1, grid_x]) / (2 * self.resolution)
        
        return np.array([dx, dy])


class StatePotential:
    """状态势场"""
    
    def __init__(self, weight: float, target_state: np.ndarray):
        self.weight = weight
        self.target_state = target_state
        
    def compute(self, state: np.ndarray) -> float:
        """计算状态势场值"""
        error = state - self.target_state
        return self.weight * np.sum(error**2)
        
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """计算状态势场梯度"""
        error = state - self.target_state
        return 2 * self.weight * error


class StateConstraint:
    """状态约束"""
    
    def __init__(self, weight: float, lower_bound: np.ndarray, upper_bound: np.ndarray):
        self.weight = weight
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def compute(self, state: np.ndarray) -> float:
        """计算约束违反成本"""
        violations = []
        
        # 下界违反
        lower_violations = np.maximum(self.lower_bound - state, 0)
        violations.extend(lower_violations)
        
        # 上界违反
        upper_violations = np.maximum(state - self.upper_bound, 0)
        violations.extend(upper_violations)
        
        return self.weight * np.sum(np.array(violations)**2)
        
    def gradient(self, state: np.ndarray) -> np.ndarray:
        """计算约束梯度"""
        grad = np.zeros_like(state)
        
        # 下界梯度
        lower_violations = self.lower_bound - state
        mask = lower_violations > 0
        grad[mask] -= 2 * self.weight * lower_violations[mask]
        
        # 上界梯度
        upper_violations = state - self.upper_bound
        mask = upper_violations > 0
        grad[mask] += 2 * self.weight * upper_violations[mask]
        
        return grad


class ControlPotential:
    """控制势场"""
    
    def __init__(self, weight: float):
        self.weight = weight
        
    def compute(self, control: np.ndarray) -> float:
        """计算控制势场值"""
        return self.weight * np.sum(control**2)
        
    def gradient(self, control: np.ndarray) -> np.ndarray:
        """计算控制势场梯度"""
        return 2 * self.weight * control


class CompositeCost(CostFunction):
    """组合成本函数"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.cost_functions = []
        
    def add_cost_function(self, cost_function: CostFunction):
        """添加成本函数"""
        self.cost_functions.append(cost_function)
        
    def compute(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> float:
        """计算总成本"""
        total_cost = 0.0
        for cost_func in self.cost_functions:
            total_cost += cost_func.compute(state, control, target)
        return total_cost
        
    def gradient(self, state: np.ndarray, control: np.ndarray,
                target: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """计算总梯度"""
        state_grad = np.zeros_like(state)
        control_grad = np.zeros_like(control)
        
        for cost_func in self.cost_functions:
            s_grad, c_grad = cost_func.gradient(state, control, target)
            state_grad += s_grad
            control_grad += c_grad
            
        return state_grad, control_grad