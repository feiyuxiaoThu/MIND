"""
iLQR优化器

实现用于MARC轨迹优化的iLQR算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.linalg import solve_banded
import warnings


class ILQROptimizer:
    """iLQR优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # iLQR参数
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.lambda_init = config.get('lambda_init', 1.0)
        self.lambda_factor = config.get('lambda_factor', 10.0)
        self.lambda_max = config.get('lambda_max', 1e6)
        self.lambda_min = config.get('lambda_min', 1e-6)
        self.delta_0 = config.get('delta_0', 2e-3)
        self.line_search_steps = config.get('line_search_steps', 10)
        self.line_search_alpha = config.get('line_search_alpha', 0.5)
        
        # 动力学参数
        self.dt = config.get('dt', 0.1)
        self.wheelbase = config.get('wheelbase', 2.5)
        
        # 控制约束
        self.control_bounds = config.get('control_bounds', {
            'acceleration': [-3.0, 3.0],
            'steering': [-0.5, 0.5]
        })
        
    def optimize(self, initial_state: np.ndarray, target_trajectory: np.ndarray,
                initial_controls: np.ndarray, dynamics_func: Optional[Callable] = None,
                cost_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        iLQR优化
        
        Args:
            initial_state: 初始状态 [x, y, v, theta, a, delta]
            target_trajectory: 目标轨迹
            initial_controls: 初始控制序列
            dynamics_func: 动力学函数 (可选)
            cost_func: 成本函数 (可选)
            
        Returns:
            optimization_result: 优化结果
        """
        horizon = len(target_trajectory)
        
        # 初始化
        controls = initial_controls.copy()
        lambda_reg = self.lambda_init
        
        # 前向传播
        trajectory = self._forward_pass(initial_state, controls, dynamics_func)
        
        # 计算初始成本
        total_cost = self._compute_total_cost(trajectory, target_trajectory, controls, cost_func)
        
        for iteration in range(self.max_iterations):
            # 后向传播
            backward_result = self._backward_pass(trajectory, target_trajectory, controls, 
                                                 lambda_reg, dynamics_func, cost_func)
            
            if not backward_result['success']:
                # 增加正则化参数
                lambda_reg *= self.lambda_factor
                lambda_reg = min(lambda_reg, self.lambda_max)
                continue
                
            k_feedforward = backward_result['k_feedforward']
            K_feedback = backward_result['K_feedback']
            expected_reduction = backward_result['expected_reduction']
            
            # 线搜索
            line_search_result = self._line_search(
                initial_state, trajectory, controls, k_feedforward, K_feedback,
                target_trajectory, dynamics_func, cost_func
            )
            
            if line_search_result['cost_reduction'] > 0:
                # 接受更新
                controls = line_search_result['new_controls']
                trajectory = line_search_result['new_trajectory']
                total_cost = line_search_result['new_cost']
                
                # 减少正则化参数
                lambda_reg *= 0.5
                lambda_reg = max(lambda_reg, self.lambda_min)
                
                # 收敛检查
                if line_search_result['cost_reduction'] < self.tolerance:
                    break
            else:
                # 拒绝更新，增加正则化参数
                lambda_reg *= self.lambda_factor
                lambda_reg = min(lambda_reg, self.lambda_max)
                
                if lambda_reg >= self.lambda_max:
                    warnings.warn("iLQR failed to converge, maximum regularization reached")
                    break
                    
        return {
            'success': True,
            'controls': controls,
            'trajectory': trajectory,
            'cost': total_cost,
            'iterations': iteration + 1,
            'converged': line_search_result['cost_reduction'] < self.tolerance
        }
        
    def _forward_pass(self, initial_state: np.ndarray, controls: np.ndarray,
                     dynamics_func: Optional[Callable] = None) -> np.ndarray:
        """
        前向传播
        
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
        
    def _backward_pass(self, trajectory: np.ndarray, target_trajectory: np.ndarray,
                      controls: np.ndarray, lambda_reg: float,
                      dynamics_func: Optional[Callable] = None,
                      cost_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        后向传播
        
        Args:
            trajectory: 状态轨迹
            target_trajectory: 目标轨迹
            controls: 控制序列
            lambda_reg: 正则化参数
            dynamics_func: 动力学函数
            cost_func: 成本函数
            
        Returns:
            backward_result: 后向传播结果
        """
        horizon = len(controls)
        state_dim = trajectory.shape[1]
        control_dim = controls.shape[1]
        
        # 初始化增益
        k_feedforward = np.zeros((horizon, control_dim))
        K_feedback = np.zeros((horizon, control_dim, state_dim))
        
        # 初始化价值函数梯度
        V_x = np.zeros(state_dim)
        V_xx = np.zeros((state_dim, state_dim))
        
        expected_reduction = 0.0
        
        # 后向传播
        for t in range(horizon - 1, -1, -1):
            # 计算成本函数梯度
            l_x, l_u, l_xx, l_uu, l_ux = self._compute_cost_gradients(
                trajectory[t], trajectory[t + 1], controls[t], 
                target_trajectory[t], cost_func
            )
            
            # 计算动力学梯度
            f_x, f_u = self._compute_dynamics_gradients(
                trajectory[t], controls[t], dynamics_func
            )
            
            # Q函数梯度
            Q_x = l_x + f_x.T @ V_x
            Q_u = l_u + f_u.T @ V_x
            Q_xx = l_xx + f_x.T @ V_xx @ f_x
            Q_uu = l_uu + f_u.T @ V_xx @ f_u
            Q_ux = l_ux + f_u.T @ V_xx @ f_x
            
            # 正则化
            Q_uu_reg = Q_uu + lambda_reg * np.eye(control_dim)
            Q_xx_reg = Q_xx + lambda_reg * np.eye(state_dim)
            
            # 检查正定性
            try:
                # Cholesky分解
                L = np.linalg.cholesky(Q_uu_reg)
                
                # 计算增益
                Q_uu_inv = np.linalg.inv(Q_uu_reg)
                k_feedforward[t] = -Q_uu_inv @ Q_u
                K_feedback[t] = -Q_uu_inv @ Q_ux
                
                # 更新价值函数梯度
                V_x = Q_x + K_feedback[t].T @ Q_uu @ k_feedforward[t]
                V_xx = Q_xx + K_feedback[t].T @ Q_uu @ K_feedback[t]
                
                # 对称化
                V_xx = 0.5 * (V_xx + V_xx.T)
                
                # 期望成本减少
                expected_reduction -= k_feedforward[t].T @ Q_u
                
            except np.linalg.LinAlgError:
                # 矩阵不是正定的
                return {
                    'success': False,
                    'reason': 'Q_uu matrix is not positive definite'
                }
                
        return {
            'success': True,
            'k_feedforward': k_feedforward,
            'K_feedback': K_feedback,
            'expected_reduction': expected_reduction
        }
        
    def _line_search(self, initial_state: np.ndarray, trajectory: np.ndarray,
                    controls: np.ndarray, k_feedforward: np.ndarray, K_feedback: np.ndarray,
                    target_trajectory: np.ndarray, dynamics_func: Optional[Callable] = None,
                    cost_func: Optional[Callable] = None) -> Dict[str, Any]:
        """
        线搜索
        
        Args:
            initial_state: 初始状态
            trajectory: 当前轨迹
            controls: 当前控制
            k_feedforward: 前馈增益
            K_feedback: 反馈增益
            target_trajectory: 目标轨迹
            dynamics_func: 动力学函数
            cost_func: 成本函数
            
        Returns:
            line_search_result: 线搜索结果
        """
        horizon = len(controls)
        alpha = 1.0
        
        # 计算当前成本
        current_cost = self._compute_total_cost(trajectory, target_trajectory, controls, cost_func)
        
        for step in range(self.line_search_steps):
            # 计算新控制
            new_controls = controls.copy()
            for t in range(horizon):
                new_controls[t] += alpha * k_feedforward[t]
                
            # 前向传播
            new_trajectory = self._forward_pass(initial_state, new_controls, dynamics_func)
            
            # 计算新成本
            new_cost = self._compute_total_cost(new_trajectory, target_trajectory, new_controls, cost_func)
            
            # Armijo条件
            if new_cost < current_cost:
                return {
                    'success': True,
                    'new_controls': new_controls,
                    'new_trajectory': new_trajectory,
                    'new_cost': new_cost,
                    'cost_reduction': current_cost - new_cost,
                    'alpha': alpha
                }
                
            # 减小步长
            alpha *= self.line_search_alpha
            
        # 线搜索失败
        return {
            'success': False,
            'new_controls': controls,
            'new_trajectory': trajectory,
            'new_cost': current_cost,
            'cost_reduction': 0.0,
            'alpha': alpha
        }
        
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
        v_next = max(0.0, v_next)
        
        x_next = x + v_next * np.cos(theta) * self.dt
        y_next = y + v_next * np.sin(theta) * self.dt
        theta_next = theta + v_next / self.wheelbase * np.tan(delta_next) * self.dt
        
        return np.array([x_next, y_next, v_next, theta_next, a_next, delta_next])
        
    def _compute_cost_gradients(self, state: np.ndarray, next_state: np.ndarray,
                               control: np.ndarray, target_state: np.ndarray,
                               cost_func: Optional[Callable] = None) -> Tuple[np.ndarray, ...]:
        """
        计算成本函数梯度
        
        Args:
            state: 当前状态
            next_state: 下一状态
            control: 控制输入
            target_state: 目标状态
            cost_func: 成本函数
            
        Returns:
            (l_x, l_u, l_xx, l_uu, l_ux): 成本函数梯度
        """
        state_dim = state.shape[0]
        control_dim = control.shape[0]
        
        if cost_func is not None:
            return cost_func(state, next_state, control, target_state)
            
        # 默认成本函数
        # 状态成本
        position_error = state[:2] - target_state[:2]
        velocity_error = state[2] - target_state[2]
        
        # 成本函数
        l = np.sum(position_error**2) + 0.1 * velocity_error**2 + 0.01 * np.sum(control**2)
        
        # 梯度
        l_x = np.zeros(state_dim)
        l_x[:2] = 2 * position_error
        l_x[2] = 0.2 * velocity_error
        
        l_u = 0.02 * control
        
        # Hessian
        l_xx = np.zeros((state_dim, state_dim))
        l_xx[0, 0] = 2.0
        l_xx[1, 1] = 2.0
        l_xx[2, 2] = 0.2
        
        l_uu = 0.02 * np.eye(control_dim)
        l_ux = np.zeros((control_dim, state_dim))
        
        return l_x, l_u, l_xx, l_uu, l_ux
        
    def _compute_dynamics_gradients(self, state: np.ndarray, control: np.ndarray,
                                   dynamics_func: Optional[Callable] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算动力学梯度
        
        Args:
            state: 当前状态
            control: 控制输入
            dynamics_func: 动力学函数
            
        Returns:
            (f_x, f_u): 动力学梯度
        """
        state_dim = state.shape[0]
        control_dim = control.shape[0]
        
        if dynamics_func is not None:
            # 数值微分
            epsilon = 1e-6
            
            f_x = np.zeros((state_dim, state_dim))
            for i in range(state_dim):
                state_plus = state.copy()
                state_plus[i] += epsilon
                state_minus = state.copy()
                state_minus[i] -= epsilon
                
                f_plus = dynamics_func(state_plus, control)
                f_minus = dynamics_func(state_minus, control)
                
                f_x[:, i] = (f_plus - f_minus) / (2 * epsilon)
                
            f_u = np.zeros((state_dim, control_dim))
            for i in range(control_dim):
                control_plus = control.copy()
                control_plus[i] += epsilon
                control_minus = control.copy()
                control_minus[i] -= epsilon
                
                f_plus = dynamics_func(state, control_plus)
                f_minus = dynamics_func(state, control_minus)
                
                f_u[:, i] = (f_plus - f_minus) / (2 * epsilon)
                
            return f_x, f_u
            
        # 解析梯度（自行车模型）
        x, y, v, theta, a, delta = state
        da, ddelta = control
        
        # 状态梯度
        f_x = np.zeros((state_dim, state_dim))
        f_x[0, 0] = 1.0
        f_x[0, 2] = np.cos(theta) * self.dt
        f_x[0, 3] = -v * np.sin(theta) * self.dt
        
        f_x[1, 1] = 1.0
        f_x[1, 2] = np.sin(theta) * self.dt
        f_x[1, 3] = v * np.cos(theta) * self.dt
        
        f_x[2, 4] = self.dt
        
        f_x[3, 2] = np.tan(delta) / self.wheelbase * self.dt
        f_x[3, 3] = 1.0
        f_x[3, 5] = v / (self.wheelbase * np.cos(delta)**2) * self.dt
        
        f_x[4, 4] = 1.0
        f_x[5, 5] = 1.0
        
        # 控制梯度
        f_u = np.zeros((state_dim, control_dim))
        f_u[4, 0] = self.dt  # da影响加速度
        f_u[5, 1] = self.dt  # ddelta影响转向角
        
        return f_x, f_u
        
    def _compute_total_cost(self, trajectory: np.ndarray, target_trajectory: np.ndarray,
                           controls: np.ndarray, cost_func: Optional[Callable] = None) -> float:
        """
        计算总成本
        
        Args:
            trajectory: 状态轨迹
            target_trajectory: 目标轨迹
            controls: 控制序列
            cost_func: 成本函数
            
        Returns:
            total_cost: 总成本
        """
        total_cost = 0.0
        horizon = len(controls)
        
        for t in range(horizon):
            if cost_func is not None:
                l_x, l_u, l_xx, l_uu, l_ux = cost_func(
                    trajectory[t], trajectory[t + 1], controls[t], target_trajectory[t]
                )
                # 这里简化处理，实际应该从成本函数计算标量值
                total_cost += np.sum(l_x**2) + np.sum(l_u**2)
            else:
                # 默认成本
                position_error = trajectory[t, :2] - target_trajectory[t, :2]
                velocity_error = trajectory[t, 2] - target_trajectory[t, 2]
                
                total_cost += np.sum(position_error**2) + 0.1 * velocity_error**2 + 0.01 * np.sum(controls[t]**2)
                
        return total_cost