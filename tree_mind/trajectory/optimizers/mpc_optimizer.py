"""
MPC优化器实现

模型预测控制器，支持硬约束处理。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from ..dynamics import DynamicsModel
from ..costs import CostFunction
from ..trajectory_tree import TrajectoryTree, TrajectoryNode, TrajectoryData, TrajectoryState, ControlInput


class MPCOptimizer:
    """MPC优化器"""
    
    def __init__(self, dynamics_model: DynamicsModel, cost_function: CostFunction,
                 config: Dict[str, Any]):
        self.dynamics = dynamics_model
        self.cost = cost_function
        self.config = config
        
        # MPC参数
        self.horizon = config.get('horizon', 20)
        self.dt = config.get('dt', 0.1)
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 约束参数
        self.max_acceleration = config.get('max_acceleration', 3.0)
        self.min_acceleration = config.get('min_acceleration', -5.0)
        self.max_steering_rate = config.get('max_steering_rate', 0.5)
        self.min_steering_rate = config.get('min_steering_rate', -0.5)
        self.max_velocity = config.get('max_velocity', 20.0)
        self.min_velocity = config.get('min_velocity', 0.0)
        
        # 状态和控制维度
        self.state_dim = 6  # [x, y, v, theta, a, delta]
        self.control_dim = 2  # [da, ddelta]
        
    def optimize(self, initial_state: np.ndarray, target_trajectory: Optional[np.ndarray] = None,
                 obstacles: Optional[List[np.ndarray]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        MPC优化
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            obstacles: 障碍物位置列表
            
        Returns:
            optimized_states: 优化后的状态序列
            optimized_controls: 优化后的控制序列
        """
        # 初始化决策变量
        initial_controls = np.zeros(self.horizon * self.control_dim)
        
        # 设置约束
        constraints = self._setup_constraints(initial_state, target_trajectory, obstacles)
        
        # 设置边界
        bounds = self._setup_bounds()
        
        # 优化
        result = minimize(
            self._objective_function,
            initial_controls,
            args=(initial_state, target_trajectory, obstacles),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            print(f"MPC optimization failed: {result.message}")
            
        # 提取优化结果
        optimized_controls = result.x.reshape(-1, self.control_dim)
        optimized_states = self._simulate_trajectory(initial_state, optimized_controls)
        
        return optimized_states, optimized_controls
        
    def _objective_function(self, control_flat: np.ndarray, initial_state: np.ndarray,
                           target_trajectory: Optional[np.ndarray] = None,
                           obstacles: Optional[List[np.ndarray]] = None) -> float:
        """目标函数"""
        controls = control_flat.reshape(-1, self.control_dim)
        states = self._simulate_trajectory(initial_state, controls)
        
        total_cost = 0.0
        
        for t in range(len(controls)):
            target = target_trajectory[t] if target_trajectory is not None else None
            
            # 基础成本
            stage_cost = self.cost.compute(states[t], controls[t], target)
            total_cost += stage_cost
            
            # 障碍物成本
            if obstacles:
                position = states[t][:2]
                for obstacle in obstacles:
                    distance = np.linalg.norm(position - obstacle)
                    if distance < 2.0:  # 安全距离
                        obstacle_cost = 100.0 * np.exp(-distance)
                        total_cost += obstacle_cost
                        
        return total_cost
        
    def _simulate_trajectory(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """模拟轨迹"""
        states = np.zeros((len(controls) + 1, self.state_dim))
        states[0] = initial_state
        
        for t, control in enumerate(controls):
            states[t + 1] = self.dynamics.step(states[t], control)
            
        return states
        
    def _setup_constraints(self, initial_state: np.ndarray, target_trajectory: Optional[np.ndarray] = None,
                          obstacles: Optional[List[np.ndarray]] = None) -> List[Dict]:
        """设置约束"""
        constraints = []
        
        # 动力学约束（隐式处理）
        def dynamics_constraint(control_flat, initial_state):
            controls = control_flat.reshape(-1, self.control_dim)
            states = self._simulate_trajectory(initial_state, controls)
            return states.flatten()  # 返回所有状态
            
        # 速度约束
        def velocity_constraint(control_flat, initial_state):
            controls = control_flat.reshape(-1, self.control_dim)
            states = self._simulate_trajectory(initial_state, controls)
            velocities = states[:, 2]
            return velocities
            
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: velocity_constraint(x, initial_state) - self.min_velocity
        })
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: self.max_velocity - velocity_constraint(x, initial_state)
        })
        
        # 终端约束（可选）
        if target_trajectory is not None:
            def terminal_constraint(control_flat, initial_state):
                controls = control_flat.reshape(-1, self.control_dim)
                states = self._simulate_trajectory(initial_state, controls)
                terminal_state = states[-1]
                terminal_target = target_trajectory[-1]
                
                # 终端位置约束
                position_error = np.linalg.norm(terminal_state[:2] - terminal_target[:2])
                velocity_error = abs(terminal_state[2] - terminal_target[2])
                
                return np.array([-position_error, -velocity_error])  # 负号表示不等式约束
                
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: terminal_constraint(x, initial_state)
            })
            
        return constraints
        
    def _setup_bounds(self) -> List[Tuple[float, float]]:
        """设置变量边界"""
        bounds = []
        
        for _ in range(self.horizon):
            # 加速度变化率边界
            bounds.append((self.min_acceleration, self.max_acceleration))
            # 转向角变化率边界
            bounds.append((self.min_steering_rate, self.max_steering_rate))
            
        return bounds
        
    def optimize_trajectory_tree(self, scenario_tree, initial_state: np.ndarray,
                                target_trajectory: np.ndarray) -> TrajectoryTree:
        """优化轨迹树"""
        trajectory_tree = TrajectoryTree(self.config)
        
        # 添加根节点
        trajectory_tree.add_root(
            TrajectoryState(
                position=initial_state[:2],
                velocity=initial_state[2],
                heading=initial_state[3],
                acceleration=initial_state[4],
                steering_angle=initial_state[5],
                timestamp=0.0
            ),
            ControlInput(acceleration=0.0, steering_rate=0.0)
        )
        
        # 获取场景分支
        scenario_branches = scenario_tree.get_scenario_branches()
        
        for branch in scenario_branches:
            # 提取场景数据
            scenario_data = branch[-1].scenario_data
            
            # 提取障碍物（从外部智能体预测）
            obstacles = []
            for exo_pred in scenario_data.exo_predictions:
                # 使用预测的终点位置作为障碍物
                obstacles.append(exo_pred.means[-1])
                
            # MPC优化
            optimized_states, optimized_controls = self.optimize(
                initial_state, target_trajectory, obstacles
            )
            
            # 构建轨迹树
            parent_id = "root"
            for t in range(1, min(len(optimized_states), self.horizon)):
                state = optimized_states[t]
                control = optimized_controls[t-1]
                
                # 计算成本
                target = target_trajectory[t] if target_trajectory is not None else None
                cost = self.cost.compute(state, control, target)
                
                # 添加轨迹节点
                trajectory_node = trajectory_tree.add_trajectory_step(
                    parent_id,
                    TrajectoryState(
                        position=state[:2],
                        velocity=state[2],
                        heading=state[3],
                        acceleration=state[4],
                        steering_angle=state[5],
                        timestamp=t * self.dt
                    ),
                    ControlInput(acceleration=control[0], steering_rate=control[1]),
                    cost,
                    scenario_data.probability
                )
                
                parent_id = trajectory_node.key
                
        return trajectory_tree
        
    def update_horizon(self, new_horizon: int):
        """更新预测视野"""
        self.horizon = new_horizon
        
    def set_constraints(self, max_acceleration: float, min_acceleration: float,
                       max_steering_rate: float, min_steering_rate: float):
        """设置约束参数"""
        self.max_acceleration = max_acceleration
        self.min_acceleration = min_acceleration
        self.max_steering_rate = max_steering_rate
        self.min_steering_rate = min_steering_rate