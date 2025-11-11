"""
动力学模型模块

实现车辆动力学模型，包括自行车模型和运动学模型。
"""

import numpy as np
from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod


class DynamicsModel(ABC):
    """动力学模型基类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.dt = config.get('dt', 0.1)
        
    @abstractmethod
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """状态转移"""
        pass
        
    @abstractmethod
    def get_jacobian(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """获取雅可比矩阵 (A, B)"""
        pass


class BicycleDynamics(DynamicsModel):
    """自行车动力学模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.wheelbase = config.get('wheelbase', 2.5)  # 轴距
        self.max_acceleration = config.get('max_acceleration', 3.0)  # 最大加速度
        self.max_deceleration = config.get('max_deceleration', 5.0)  # 最大减速度
        self.max_steering_angle = config.get('max_steering_angle', 0.5)  # 最大转向角
        self.max_steering_rate = config.get('max_steering_rate', 0.5)  # 最大转向速率
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        状态转移
        
        Args:
            state: [x, y, v, theta, a, delta] 位置x,y, 速度v, 航向角theta, 加速度a, 转向角delta
            control: [da, ddelta] 加速度变化率da, 转向角变化率ddelta
            
        Returns:
            next_state: 下一时刻状态
        """
        x, y, v, theta, a, delta = state
        da, ddelta = control
        
        # 更新加速度和转向角
        a_next = a + da * self.dt
        delta_next = delta + ddelta * self.dt
        
        # 限制加速度和转向角
        a_next = np.clip(a_next, -self.max_deceleration, self.max_acceleration)
        delta_next = np.clip(delta_next, -self.max_steering_angle, self.max_steering_angle)
        
        # 更新速度
        v_next = v + a_next * self.dt
        v_next = max(0.0, v_next)  # 速度非负
        
        # 更新位置和航向
        x_next = x + v_next * np.cos(theta) * self.dt
        y_next = y + v_next * np.sin(theta) * self.dt
        theta_next = theta + v_next / self.wheelbase * np.tan(delta_next) * self.dt
        
        # 归一化航向角
        theta_next = self._normalize_angle(theta_next)
        
        return np.array([x_next, y_next, v_next, theta_next, a_next, delta_next])
        
    def get_jacobian(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算雅可比矩阵
        
        Returns:
            A: 状态矩阵 df/dx
            B: 控制矩阵 df/du
        """
        x, y, v, theta, a, delta = state
        da, ddelta = control
        
        # 状态矩阵 A = df/dx
        A = np.zeros((6, 6))
        A[0, 2] = np.cos(theta) * self.dt  # dx/dv
        A[0, 3] = -v * np.sin(theta) * self.dt  # dx/dtheta
        A[1, 2] = np.sin(theta) * self.dt  # dy/dv
        A[1, 3] = v * np.cos(theta) * self.dt  # dy/dtheta
        A[2, 4] = self.dt  # dv/da
        A[3, 2] = np.tan(delta) / self.wheelbase * self.dt  # dtheta/dv
        A[3, 5] = v / (self.wheelbase * np.cos(delta)**2) * self.dt  # dtheta/ddelta
        A[4, 4] = 1.0  # da/da
        A[5, 5] = 1.0  # ddelta/ddelta
        
        # 控制矩阵 B = df/du
        B = np.zeros((6, 2))
        B[4, 0] = self.dt  # da/da_control
        B[5, 1] = self.dt  # ddelta/ddelta_control
        
        return A, B
        
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def validate_state(self, state: np.ndarray) -> bool:
        """验证状态是否合理"""
        x, y, v, theta, a, delta = state
        
        # 检查速度
        if v < 0 or v > 30.0:  # 最大速度30m/s
            return False
            
        # 检查加速度
        if a < -self.max_deceleration or a > self.max_acceleration:
            return False
            
        # 检查转向角
        if abs(delta) > self.max_steering_angle:
            return False
            
        return True
        
    def validate_control(self, control: np.ndarray) -> bool:
        """验证控制输入是否合理"""
        da, ddelta = control
        
        # 检查加速度变化率
        if abs(da) > 10.0:  # 最大加速度变化率
            return False
            
        # 检查转向角变化率
        if abs(ddelta) > self.max_steering_rate:
            return False
            
        return True


class KinematicsModel(DynamicsModel):
    """简化的运动学模型"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.max_acceleration = config.get('max_acceleration', 3.0)
        self.max_deceleration = config.get('max_deceleration', 5.0)
        self.max_steering_angle = config.get('max_steering_angle', 0.5)
        self.max_steering_rate = config.get('max_steering_rate', 0.5)
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        简化的运动学模型
        
        Args:
            state: [x, y, v, theta] 位置x,y, 速度v, 航向角theta
            control: [a, delta] 加速度a, 转向角delta
            
        Returns:
            next_state: 下一时刻状态
        """
        x, y, v, theta = state
        a, delta = control
        
        # 限制控制输入
        a = np.clip(a, -self.max_deceleration, self.max_acceleration)
        delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        
        # 更新速度
        v_next = v + a * self.dt
        v_next = max(0.0, v_next)
        
        # 更新位置和航向
        x_next = x + v_next * np.cos(theta) * self.dt
        y_next = y + v_next * np.sin(theta) * self.dt
        theta_next = theta + v_next * np.tan(delta) / 2.5 * self.dt  # 假设轴距2.5m
        
        # 归一化航向角
        theta_next = self._normalize_angle(theta_next)
        
        return np.array([x_next, y_next, v_next, theta_next])
        
    def get_jacobian(self, state: np.ndarray, control: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """计算雅可比矩阵"""
        x, y, v, theta = state
        a, delta = control
        
        # 状态矩阵 A
        A = np.zeros((4, 4))
        A[0, 2] = np.cos(theta) * self.dt
        A[0, 3] = -v * np.sin(theta) * self.dt
        A[1, 2] = np.sin(theta) * self.dt
        A[1, 3] = v * np.cos(theta) * self.dt
        A[2, 2] = 1.0
        A[3, 2] = np.tan(delta) / 2.5 * self.dt
        A[3, 3] = 1.0
        
        # 控制矩阵 B
        B = np.zeros((4, 2))
        B[2, 0] = self.dt
        B[3, 1] = v / (2.5 * np.cos(delta)**2) * self.dt
        
        return A, B
        
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到[-π, π]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


class ExtendedBicycleDynamics(BicycleDynamics):
    """扩展的自行车动力学模型，考虑轮胎侧偏刚度"""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.mass = config.get('mass', 1500.0)  # 车辆质量
        self.Iz = config.get('Iz', 2500.0)  # 转动惯量
        self.lf = config.get('lf', 1.2)  # 前轴到质心距离
        self.lr = config.get('lr', 1.3)  # 后轴到质心距离
        self.Cf = config.get('Cf', 55000.0)  # 前轮胎侧偏刚度
        self.Cr = config.get('Cr', 60000.0)  # 后轮胎侧偏刚度
        
    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        扩展自行车模型状态转移
        
        Args:
            state: [x, y, v, theta, beta, r] 位置x,y, 速度v, 航向角theta, 侧偏角beta, 横摆角速度r
            control: [a, delta] 加速度a, 前轮转角delta
            
        Returns:
            next_state: 下一时刻状态
        """
        x, y, v, theta, beta, r = state
        a, delta = control
        
        # 限制控制输入
        a = np.clip(a, -self.max_deceleration, self.max_acceleration)
        delta = np.clip(delta, -self.max_steering_angle, self.max_steering_angle)
        
        # 计算侧偏刚度相关参数
        alpha_f = delta - beta - self.lf * r / v  # 前轮胎侧偏角
        alpha_r = -beta + self.lr * r / v  # 后轮胎侧偏角
        
        # 计算力和力矩
        Fyf = 2 * self.Cf * alpha_f  # 前轮胎侧向力
        Fyr = 2 * self.Cr * alpha_r  # 后轮胎侧向力
        
        # 状态导数
        x_dot = v * np.cos(theta + beta)
        y_dot = v * np.sin(theta + beta)
        v_dot = a
        theta_dot = r
        beta_dot = (Fyf + Fyr) / (self.mass * v) - r
        r_dot = (self.lf * Fyf - self.lr * Fyr) / self.Iz
        
        # 积分
        x_next = x + x_dot * self.dt
        y_next = y + y_dot * self.dt
        v_next = v + v_dot * self.dt
        v_next = max(0.1, v_next)  # 避免除零
        theta_next = theta + theta_dot * self.dt
        beta_next = beta + beta_dot * self.dt
        r_next = r + r_dot * self.dt
        
        # 归一化角度
        theta_next = self._normalize_angle(theta_next)
        beta_next = np.clip(beta_next, -0.5, 0.5)  # 限制侧偏角
        
        return np.array([x_next, y_next, v_next, theta_next, beta_next, r_next])