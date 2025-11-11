"""
iLQR优化器实现

迭代线性二次调节器，用于轨迹优化。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..dynamics import DynamicsModel
from ..costs import CostFunction
from ..trajectory_tree import TrajectoryTree, TrajectoryNode, TrajectoryData, TrajectoryState, ControlInput


class ILQROptimizer:
    """iLQR优化器"""
    
    def __init__(self, dynamics_model: DynamicsModel, cost_function: CostFunction, 
                 config: Dict[str, Any]):
        self.dynamics = dynamics_model
        self.cost = cost_function
        self.config = config
        
        # iLQR参数
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.regularization = config.get('regularization', 1e-4)
        self.alpha = config.get('alpha', 1.0)  # 线搜索参数
        self.max_line_search = config.get('max_line_search', 10)
        
        # 状态和控制维度
        self.state_dim = 6  # [x, y, v, theta, a, delta]
        self.control_dim = 2  # [da, ddelta]
        
    def optimize(self, initial_state: np.ndarray, initial_controls: np.ndarray,
                horizon: int, target_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        iLQR优化
        
        Args:
            initial_state: 初始状态
            initial_controls: 初始控制序列
            horizon: 优化视野
            target_trajectory: 目标轨迹
            
        Returns:
            optimized_states: 优化后的状态序列
            optimized_controls: 优化后的控制序列
        """
        # 初始化
        states = self._forward_rollout(initial_state, initial_controls)
        controls = initial_controls.copy()
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 线性化
            A, B = self._linearize_dynamics(states, controls)
            
            # 二次化成本
            Q, R, q, r = self._quadraticize_cost(states, controls, target_trajectory)
            
            # 后向传递
            K, k = self._backward_pass(A, B, Q, R, q, r)
            
            # 前向传递和线搜索
            new_states, new_controls, cost_reduction = self._forward_pass(
                states, controls, K, k, target_trajectory
            )
            
            # 检查收敛
            if cost_reduction < self.tolerance:
                print(f"iLQR converged at iteration {iteration}")
                break
                
            # 更新
            states = new_states
            controls = new_controls
            
        return states, controls
        
    def _forward_rollout(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """前向模拟"""
        states = np.zeros((len(controls) + 1, self.state_dim))
        states[0] = initial_state
        
        for t, control in enumerate(controls):
            states[t + 1] = self.dynamics.step(states[t], control)
            
        return states
        
    def _linearize_dynamics(self, states: np.ndarray, controls: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """线性化动力学模型"""
        horizon = len(controls)
        A = np.zeros((horizon, self.state_dim, self.state_dim))
        B = np.zeros((horizon, self.state_dim, self.control_dim))
        
        for t in range(horizon):
            A[t], B[t] = self.dynamics.get_jacobian(states[t], controls[t])
            
        return A, B
        
    def _quadraticize_cost(self, states: np.ndarray, controls: np.ndarray,
                          target_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """二次化成本函数"""
        horizon = len(controls)
        
        Q = np.zeros((horizon, self.state_dim, self.state_dim))
        R = np.zeros((horizon, self.control_dim, self.control_dim))
        q = np.zeros((horizon, self.state_dim))
        r = np.zeros((horizon, self.control_dim))
        
        for t in range(horizon):
            target = target_trajectory[t] if target_trajectory is not None else None
            
            # 使用数值方法计算Hessian和梯度
            state_grad, control_grad = self.cost.gradient(states[t], controls[t], target)
            
            # 近似Hessian（对角矩阵）
            state_hessian = np.eye(self.state_dim) * 0.1  # 简化处理
            control_hessian = np.eye(self.control_dim) * 0.1
            
            Q[t] = state_hessian
            R[t] = control_hessian
            q[t] = state_grad
            r[t] = control_grad
            
        return Q, R, q, r
        
    def _backward_pass(self, A: np.ndarray, B: np.ndarray, Q: np.ndarray, R: np.ndarray,
                      q: np.ndarray, r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """后向传递计算增益"""
        horizon = len(A)
        
        K = np.zeros((horizon, self.control_dim, self.state_dim))
        k = np.zeros((horizon, self.control_dim))
        
        # 初始化终端条件
        P = Q[-1]
        p = q[-1]
        
        for t in range(horizon - 1, -1, -1):
            # 计算增益
            Q_uu = R[t] + B[t].T @ P @ B[t]
            Q_uu_reg = Q_uu + self.regularization * np.eye(self.control_dim)
            
            # 修复维度问题：Q_ux应该是 (control_dim, state_dim)
            Q_ux = B[t].T @ P @ A[t]
            Q_xx = A[t].T @ P @ A[t] + Q[t]
            Q_x = A[t].T @ p + q[t]
            
            # 计算反馈增益和前馈增益
            K[t] = -np.linalg.solve(Q_uu_reg, Q_ux)
            
            # 修复维度问题：Q_u 应该是控制维度的梯度
            Q_u = r[t] + B[t].T @ p  # 控制梯度，维度为 control_dim
            k[t] = -np.linalg.solve(Q_uu_reg, Q_u).flatten()
            
            # 更新P和p
            P = Q_xx + K[t].T @ Q_uu @ K[t]
            p = Q_x + K[t].T @ Q_uu @ k[t]
            
        return K, k
        
    def _forward_pass(self, states: np.ndarray, controls: np.ndarray, K: np.ndarray, k: np.ndarray,
                     target_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """前向传递和线搜索"""
        horizon = len(controls)
        
        # 计算原始成本
        original_cost = self._compute_total_cost(states, controls, target_trajectory)
        
        # 尝试不同的步长
        for alpha in [1.0, 0.5, 0.25, 0.125, 0.0625]:
            new_states = np.zeros_like(states)
            new_controls = np.zeros_like(controls)
            new_states[0] = states[0]
            
            # 应用控制律
            for t in range(horizon):
                delta_u = K[t] @ (new_states[t] - states[t]) + k[t]
                new_controls[t] = controls[t] + alpha * delta_u
                new_states[t + 1] = self.dynamics.step(new_states[t], new_controls[t])
                
            # 计算新成本
            new_cost = self._compute_total_cost(new_states, new_controls, target_trajectory)
            cost_reduction = original_cost - new_cost
            
            if cost_reduction > 0:
                return new_states, new_controls, cost_reduction
                
        # 如果没有改进，返回原始解
        return states, controls, 0.0
        
    def _compute_total_cost(self, states: np.ndarray, controls: np.ndarray,
                           target_trajectory: Optional[np.ndarray] = None) -> float:
        """计算总成本"""
        total_cost = 0.0
        
        for t in range(len(controls)):
            target = target_trajectory[t] if target_trajectory is not None else None
            total_cost += self.cost.compute(states[t], controls[t], target)
            
        return total_cost
        
    def optimize_trajectory_tree(self, scenario_tree, initial_state: np.ndarray,
                                target_trajectory: np.ndarray) -> TrajectoryTree:
        """优化轨迹树"""
        trajectory_tree = TrajectoryTree(self.config)
        
        # 添加根节点
        initial_control = np.array([0.0, 0.0])  # 初始控制
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
            
            # 初始化控制序列
            horizon = len(scenario_data.ego_prediction.means)
            initial_controls = np.zeros((horizon, self.control_dim))
            
            # 优化每条分支
            optimized_states, optimized_controls = self.optimize(
                initial_state, initial_controls, horizon, target_trajectory
            )
            
            # 构建轨迹树
            parent_id = "root"
            for t in range(1, len(optimized_states)):
                state = optimized_states[t]
                control = optimized_controls[t-1]
                
                # 计算成本 - 修复索引越界问题
                target_idx = min(t, len(target_trajectory) - 1) if target_trajectory is not None else None
                target = target_trajectory[target_idx] if target_trajectory is not None else None
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
                        timestamp=t * self.dynamics.dt
                    ),
                    ControlInput(acceleration=control[0], steering_rate=control[1]),
                    cost,
                    scenario_data.probability
                )
                
                parent_id = trajectory_node.key
                
        return trajectory_tree