"""
MPC优化器

实现用于MARC轨迹优化的模型预测控制算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings


class MPCOptimizer:
    """模型预测控制优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # MPC参数
        self.horizon = config.get('horizon', 20)
        self.dt = config.get('dt', 0.1)
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 权重参数
        self.state_weights = config.get('state_weights', np.array([10.0, 10.0, 1.0, 0.1, 0.1, 0.1]))
        self.control_weights = config.get('control_weights', np.array([1.0, 1.0]))
        self.terminal_weights = config.get('terminal_weights', np.array([100.0, 100.0, 10.0, 1.0, 0.1, 0.1]))
        
        # 动力学参数
        self.wheelbase = config.get('wheelbase', 2.5)
        
        # 控制约束
        self.control_bounds = config.get('control_bounds', {
            'acceleration': [-3.0, 3.0],
            'steering': [-0.5, 0.5]
        })
        
        # 状态约束
        self.state_bounds = config.get('state_bounds', {
            'velocity': [0.0, 30.0],
            'acceleration': [-5.0, 5.0]
        })
        
    def optimize(self, initial_state: np.ndarray, target_trajectory: np.ndarray,
                obstacles: List[Dict[str, Any]] = None,
                dynamics_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        MPC优化
        
        Args:
            initial_state: 初始状态 [x, y, v, theta, a, delta]
            target_trajectory: 目标轨迹
            obstacles: 障碍物列表
            dynamics_func: 动力学函数 (可选)
            
        Returns:
            optimization_result: 优化结果
        """
        # 确定预测范围
        prediction_horizon = min(self.horizon, len(target_trajectory))
        
        # 初始化控制序列
        initial_controls = self._initialize_controls(initial_state, target_trajectory[:prediction_horizon])
        
        # 定义优化变量
        num_controls = 2  # [da, ddelta]
        variable_size = prediction_horizon * num_controls
        
        # 初始猜测
        x0 = initial_controls.flatten()
        
        # 定义约束
        constraints = self._define_constraints(initial_state, prediction_horizon, 
                                            obstacles, dynamics_func)
        
        # 定义边界
        bounds = self._define_bounds(prediction_horizon, num_controls)
        
        # 优化
        try:
            result = minimize(
                lambda x: self._objective_function(x, initial_state, target_trajectory[:prediction_horizon], 
                                                 dynamics_func),
                x0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={
                    'maxiter': self.max_iterations,
                    'ftol': self.tolerance,
                    'disp': False
                }
            )
            
            if not result.success:
                return {
                    'success': False,
                    'reason': result.message,
                    'controls': initial_controls,
                    'trajectory': self._simulate_trajectory(initial_state, initial_controls, dynamics_func)
                }
                
            # 提取结果
            optimized_controls = result.x.reshape(prediction_horizon, num_controls)
            
            # 模拟轨迹
            trajectory = self._simulate_trajectory(initial_state, optimized_controls, dynamics_func)
            
            return {
                'success': True,
                'controls': optimized_controls,
                'trajectory': trajectory,
                'cost': result.fun,
                'iterations': result.nit
            }
            
        except Exception as e:
            return {
                'success': False,
                'reason': f'Optimization failed: {str(e)}',
                'controls': initial_controls,
                'trajectory': self._simulate_trajectory(initial_state, initial_controls, dynamics_func)
            }
            
    def _initialize_controls(self, initial_state: np.ndarray, 
                           target_trajectory: np.ndarray) -> np.ndarray:
        """
        初始化控制序列
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            
        Returns:
            initial_controls: 初始控制序列
        """
        horizon = len(target_trajectory)
        controls = np.zeros((horizon, 2))
        
        for t in range(horizon):
            if t == 0:
                current_state = initial_state
            else:
                # 使用前一时刻的状态
                current_state = self._default_dynamics(current_state, controls[t-1])
                
            target_state = target_trajectory[t]
            
            # 计算朝向目标的控制
            dx = target_state[0] - current_state[0]
            dy = target_state[1] - current_state[1]
            
            # 速度控制
            target_velocity = target_state[2]
            current_velocity = current_state[2]
            velocity_error = target_velocity - current_velocity
            
            controls[t, 0] = np.clip(velocity_error * 2.0, 
                                    self.control_bounds['acceleration'][0],
                                    self.control_bounds['acceleration'][1])
            
            # 转向控制
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                target_angle = np.arctan2(dy, dx)
                current_angle = current_state[3]
                angle_error = self._normalize_angle(target_angle - current_angle)
                
                controls[t, 1] = np.clip(angle_error * 2.0,
                                       self.control_bounds['steering'][0],
                                       self.control_bounds['steering'][1])
            else:
                controls[t, 1] = 0.0
                
        return controls
        
    def _objective_function(self, control_flat: np.ndarray, initial_state: np.ndarray,
                           target_trajectory: np.ndarray,
                           dynamics_func: Optional[Callable] = None) -> float:
        """
        目标函数
        
        Args:
            control_flat: 扁平化的控制序列
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            dynamics_func: 动力学函数
            
        Returns:
            cost: 总成本
        """
        horizon = len(target_trajectory)
        controls = control_flat.reshape(horizon, 2)
        
        # 模拟轨迹
        trajectory = self._simulate_trajectory(initial_state, controls, dynamics_func)
        
        # 计算成本
        cost = 0.0
        
        for t in range(horizon):
            # 状态成本
            state_error = trajectory[t] - target_trajectory[t]
            state_cost = np.sum(self.state_weights * state_error**2)
            cost += state_cost
            
            # 控制成本
            control_cost = np.sum(self.control_weights * controls[t]**2)
            cost += control_cost
            
            # 控制变化成本（平滑性）
            if t > 0:
                control_change = controls[t] - controls[t-1]
                smoothness_cost = np.sum(control_change**2) * 0.1
                cost += smoothness_cost
                
        # 终端成本
        terminal_error = trajectory[horizon] - target_trajectory[-1]
        terminal_cost = np.sum(self.terminal_weights * terminal_error**2)
        cost += terminal_cost
        
        return cost
        
    def _simulate_trajectory(self, initial_state: np.ndarray, controls: np.ndarray,
                           dynamics_func: Optional[Callable] = None) -> np.ndarray:
        """
        模拟轨迹
        
        Args:
            initial_state: 初始状态
            controls: 控制序列
            dynamics_func: 动力学函数
            
        Returns:
            trajectory: 状态轨迹
        """
        horizon = len(controls)
        trajectory = np.zeros((horizon + 1, initial_state.shape[0]))
        trajectory[0] = initial_state
        
        for t in range(horizon):
            if dynamics_func is not None:
                trajectory[t + 1] = dynamics_func(trajectory[t], controls[t])
            else:
                trajectory[t + 1] = self._default_dynamics(trajectory[t], controls[t])
                
        return trajectory
        
    def _default_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """
        默认动力学模型（自行车模型）
        
        Args:
            state: 当前状态 [x, y, v, theta, a, delta]
            control: 控制输入 [da, ddelta]
            
        Returns:
            next_state: 下一状态
        """
        x, y, v, theta, a, delta = state
        da, ddelta = control
        
        # 更新控制
        a_next = np.clip(a + da * self.dt, self.control_bounds['acceleration'][0], 
                        self.control_bounds['acceleration'][1])
        delta_next = np.clip(delta + ddelta * self.dt, self.control_bounds['steering'][0], 
                            self.control_bounds['steering'][1])
        
        # 更新状态
        v_next = v + a_next * self.dt
        v_next = np.clip(v_next, self.state_bounds['velocity'][0], 
                        self.state_bounds['velocity'][1])
        
        x_next = x + v_next * np.cos(theta) * self.dt
        y_next = y + v_next * np.sin(theta) * self.dt
        theta_next = theta + v_next / self.wheelbase * np.tan(delta_next) * self.dt
        
        return np.array([x_next, y_next, v_next, theta_next, a_next, delta_next])
        
    def _define_constraints(self, initial_state: np.ndarray, horizon: int,
                          obstacles: List[Dict[str, Any]] = None,
                          dynamics_func: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """
        定义约束条件
        
        Args:
            initial_state: 初始状态
            horizon: 预测范围
            obstacles: 障碍物列表
            dynamics_func: 动力学函数
            
        Returns:
            constraints: 约束列表
        """
        constraints = []
        
        # 避障约束
        if obstacles:
            for t in range(horizon):
                for i, obstacle in enumerate(obstacles):
                    def avoid_obstacle_constraint(control_flat, t=t, obstacle=obstacle):
                        controls = control_flat.reshape(horizon, 2)
                        trajectory = self._simulate_trajectory(initial_state, controls, dynamics_func)
                        
                        ego_pos = trajectory[t, :2]
                        obstacle_pos = np.array([obstacle['x'], obstacle['y']])
                        obstacle_radius = obstacle.get('radius', 2.0)
                        
                        distance = np.linalg.norm(ego_pos - obstacle_pos)
                        
                        return distance - obstacle_radius - 1.0  # 安全距离
                        
                    constraints.append({
                        'type': 'ineq',
                        'fun': avoid_obstacle_constraint
                    })
                    
        # 速度约束
        def velocity_constraint(control_flat):
            controls = control_flat.reshape(horizon, 2)
            trajectory = self._simulate_trajectory(initial_state, controls, dynamics_func)
            
            velocities = trajectory[:, 2]
            
            # 返回速度约束 violations
            min_violation = self.state_bounds['velocity'][0] - velocities
            max_violation = velocities - self.state_bounds['velocity'][1]
            
            return np.concatenate([min_violation, max_violation])
            
        constraints.append({
            'type': 'ineq',
            'fun': velocity_constraint
        })
        
        return constraints
        
    def _define_bounds(self, horizon: int, num_controls: int) -> Bounds:
        """
        定义变量边界
        
        Args:
            horizon: 预测范围
            num_controls: 控制维度
            
        Returns:
            bounds: 变量边界
        """
        lower_bounds = []
        upper_bounds = []
        
        for t in range(horizon):
            lower_bounds.extend([
                self.control_bounds['acceleration'][0],  # da
                self.control_bounds['steering'][0]       # ddelta
            ])
            upper_bounds.extend([
                self.control_bounds['acceleration'][1],  # da
                self.control_bounds['steering'][1]       # ddelta
            ])
            
        return Bounds(lower_bounds, upper_bounds)
        
    def _normalize_angle(self, angle: float) -> float:
        """
        归一化角度到[-pi, pi]
        
        Args:
            angle: 角度值
            
        Returns:
            normalized_angle: 归一化角度
        """
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def update_reference(self, new_target_trajectory: np.ndarray) -> None:
        """
        更新参考轨迹
        
        Args:
            new_target_trajectory: 新的目标轨迹
        """
        self.reference_trajectory = new_target_trajectory
        
    def add_obstacle(self, obstacle: Dict[str, Any]) -> None:
        """
        添加障碍物
        
        Args:
            obstacle: 障碍物信息 {'x': float, 'y': float, 'radius': float}
        """
        if not hasattr(self, 'obstacles'):
            self.obstacles = []
        self.obstacles.append(obstacle)
        
    def clear_obstacles(self) -> None:
        """清除所有障碍物"""
        if hasattr(self, 'obstacles'):
            self.obstacles.clear()
            
    def predict_trajectory(self, initial_state: np.ndarray, 
                          controls: np.ndarray,
                          dynamics_func: Optional[Callable] = None) -> np.ndarray:
        """
        预测轨迹
        
        Args:
            initial_state: 初始状态
            controls: 控制序列
            dynamics_func: 动力学函数
            
        Returns:
            predicted_trajectory: 预测轨迹
        """
        return self._simulate_trajectory(initial_state, controls, dynamics_func)
        
    def evaluate_cost(self, trajectory: np.ndarray, controls: np.ndarray,
                     target_trajectory: np.ndarray) -> float:
        """
        评估成本
        
        Args:
            trajectory: 状态轨迹
            controls: 控制序列
            target_trajectory: 目标轨迹
            
        Returns:
            cost: 总成本
        """
        horizon = len(controls)
        cost = 0.0
        
        for t in range(horizon):
            # 状态成本
            state_error = trajectory[t] - target_trajectory[t]
            state_cost = np.sum(self.state_weights * state_error**2)
            cost += state_cost
            
            # 控制成本
            control_cost = np.sum(self.control_weights * controls[t]**2)
            cost += control_cost
            
            # 控制变化成本
            if t > 0:
                control_change = controls[t] - controls[t-1]
                smoothness_cost = np.sum(control_change**2) * 0.1
                cost += smoothness_cost
                
        # 终端成本
        terminal_error = trajectory[horizon] - target_trajectory[-1]
        terminal_cost = np.sum(self.terminal_weights * terminal_error**2)
        cost += terminal_cost
        
        return cost