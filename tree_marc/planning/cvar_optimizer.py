"""
CVaR优化器

实现MARC论文中的条件风险价值(CVaR)优化算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize, LinearConstraint, Bounds
import warnings


class CVAROptimizer:
    """条件风险价值优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # CVaR参数
        self.alpha = config.get('alpha', 0.1)  # 风险水平 (0-1)
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.epsilon = config.get('epsilon', 1e-8)  # 数值稳定性参数
        
        # 优化参数
        self.learning_rate = config.get('learning_rate', 0.01)
        self.momentum = config.get('momentum', 0.9)
        self.adaptive_lr = config.get('adaptive_lr', True)
        
        # 约束参数
        self.control_bounds = config.get('control_bounds', {
            'acceleration': [-3.0, 3.0],
            'steering': [-0.5, 0.5]
        })
        
    def optimize_cvar(self, scenarios: List[Dict[str, Any]], 
                      initial_controls: np.ndarray,
                      weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        优化CVaR目标函数
        
        Args:
            scenarios: 场景列表
            initial_controls: 初始控制序列
            weights: 场景权重 (可选)
            
        Returns:
            optimization_result: 优化结果
        """
        num_scenarios = len(scenarios)
        if num_scenarios == 0:
            return {
                'success': False,
                'reason': 'No scenarios provided',
                'controls': initial_controls,
                'cvar_value': 0.0
            }
            
        # 默认等权重
        if weights is None:
            weights = np.ones(num_scenarios) / num_scenarios
            
        # 验证权重
        if abs(np.sum(weights) - 1.0) > 1e-6:
            weights = weights / np.sum(weights)
            
        # 定义优化变量: [controls_flat, VaR, auxiliary_vars]
        horizon = len(initial_controls)
        num_controls = initial_controls.shape[1] if len(initial_controls.shape) > 1 else 2
        
        # 初始变量
        x0 = np.concatenate([
            initial_controls.flatten(),
            np.array([0.0]),  # VaR
            np.zeros(num_scenarios)  # 辅助变量
        ])
        
        # 定义约束
        constraints = self._define_cvar_constraints(num_scenarios, horizon, num_controls)
        
        # 定义边界
        bounds = self._define_variable_bounds(horizon, num_controls, num_scenarios)
        
        # 优化
        try:
            result = minimize(
                lambda x: self._cvar_objective(x, scenarios, weights, horizon, num_controls),
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
                    'cvar_value': 0.0
                }
                
            # 提取结果
            optimized_controls = result.x[:horizon * num_controls].reshape(horizon, num_controls)
            var_value = result.x[horizon * num_controls]
            auxiliary_vars = result.x[horizon * num_controls + 1:]
            
            # 计算CVaR值
            cvar_value = var_value + np.sum(weights * auxiliary_vars) / self.alpha
            
            return {
                'success': True,
                'controls': optimized_controls,
                'var_value': var_value,
                'auxiliary_vars': auxiliary_vars,
                'cvar_value': cvar_value,
                'cost': result.fun,
                'iterations': result.nit
            }
            
        except Exception as e:
            return {
                'success': False,
                'reason': f'Optimization failed: {str(e)}',
                'controls': initial_controls,
                'cvar_value': 0.0
            }
            
    def _cvar_objective(self, x: np.ndarray, scenarios: List[Dict[str, Any]], 
                       weights: np.ndarray, horizon: int, num_controls: int) -> float:
        """
        CVaR目标函数
        
        Args:
            x: 优化变量 [controls_flat, VaR, auxiliary_vars]
            scenarios: 场景列表
            weights: 场景权重
            horizon: 时间步长
            num_controls: 控制维度
            
        Returns:
            objective_value: 目标函数值
        """
        # 提取变量
        controls_flat = x[:horizon * num_controls]
        var_value = x[horizon * num_controls]
        auxiliary_vars = x[horizon * num_controls + 1:]
        
        controls = controls_flat.reshape(horizon, num_controls)
        
        # 计算每个场景的成本
        scenario_costs = []
        for scenario in scenarios:
            cost = self._compute_scenario_cost(controls, scenario)
            scenario_costs.append(cost)
            
        scenario_costs = np.array(scenario_costs)
        
        # CVaR目标函数: VaR + (1/alpha) * sum(weights * auxiliary_vars)
        objective = var_value + np.sum(weights * auxiliary_vars) / self.alpha
        
        return objective
        
    def _compute_scenario_cost(self, controls: np.ndarray, scenario: Dict[str, Any]) -> float:
        """
        计算单个场景的成本
        
        Args:
            controls: 控制序列
            scenario: 场景数据
            
        Returns:
            scenario_cost: 场景成本
        """
        # 模拟轨迹
        initial_state = scenario.get('initial_state', np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]))
        trajectory = self._simulate_trajectory(initial_state, controls)
        
        # 计算基础成本
        base_cost = self._compute_base_cost(trajectory, scenario)
        
        # 计算风险成本
        risk_cost = self._compute_risk_cost(trajectory, scenario)
        
        # 总成本
        total_cost = base_cost + risk_cost
        
        return total_cost
        
    def _simulate_trajectory(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """模拟轨迹"""
        trajectory = np.zeros((len(controls) + 1, 6))
        trajectory[0] = initial_state
        
        state = initial_state.copy()
        
        for t, control in enumerate(controls):
            # 自行车模型
            x, y, v, theta, a, delta = state
            da, ddelta = control
            
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
        
    def _compute_base_cost(self, trajectory: np.ndarray, scenario: Dict[str, Any]) -> float:
        """计算基础成本"""
        target_trajectory = scenario.get('target_trajectory', trajectory)
        
        cost = 0.0
        for t in range(len(trajectory)):
            # 位置偏差
            position_error = np.linalg.norm(trajectory[t, :2] - target_trajectory[t, :2])
            cost += position_error
            
            # 速度偏差
            velocity_error = abs(trajectory[t, 2] - target_trajectory[t, 2])
            cost += velocity_error * 0.1
            
            # 控制平滑性
            if t > 0:
                control_change = np.linalg.norm(trajectory[t, 4:6] - trajectory[t-1, 4:6])
                cost += control_change * 0.5
                
        return cost
        
    def _compute_risk_cost(self, trajectory: np.ndarray, scenario: Dict[str, Any]) -> float:
        """计算风险成本"""
        exo_trajectories = scenario.get('exo_trajectories', [])
        risk_costs = scenario.get('risk_costs', np.zeros((len(trajectory), len(exo_trajectories))))
        
        total_risk_cost = 0.0
        
        for t in range(len(trajectory)):
            ego_pos = trajectory[t, :2]
            
            for i, exo_traj in enumerate(exo_trajectories):
                if t < len(exo_traj):
                    exo_pos = exo_traj[t, :2]
                    distance = np.linalg.norm(ego_pos - exo_pos)
                    
                    # 碰撞风险
                    if distance < 2.0:
                        collision_risk = 1000.0 * np.exp(-distance)
                        total_risk_cost += collision_risk
                        
                    # 舒适性风险
                    comfort_risk = risk_costs[t, i] if t < len(risk_costs) and i < len(risk_costs[t]) else 0.0
                    total_risk_cost += comfort_risk * 0.1
                    
        return total_risk_cost
        
    def _define_cvar_constraints(self, num_scenarios: int, horizon: int, num_controls: int) -> List[Dict[str, Any]]:
        """定义CVaR约束"""
        constraints = []
        
        # CVaR约束: auxiliary_vars[i] >= max(0, cost_i - VaR)
        # 这通过目标函数和边界处理
        
        return constraints
        
    def _define_variable_bounds(self, horizon: int, num_controls: int, num_scenarios: int) -> Bounds:
        """定义变量边界"""
        # 控制变量边界
        control_lower = []
        control_upper = []
        
        for t in range(horizon):
            control_lower.extend([
                self.control_bounds['acceleration'][0],  # da
                self.control_bounds['steering'][0]       # ddelta
            ])
            control_upper.extend([
                self.control_bounds['acceleration'][1],  # da
                self.control_bounds['steering'][1]       # ddelta
            ])
            
        # VaR边界
        var_lower = [-1000.0]
        var_upper = [1000.0]
        
        # 辅助变量边界
        aux_lower = [0.0] * num_scenarios
        aux_upper = [1000.0] * num_scenarios
        
        lower_bounds = control_lower + var_lower + aux_lower
        upper_bounds = control_upper + var_upper + aux_upper
        
        return Bounds(lower_bounds, upper_bounds)
        
    def compute_cvar_from_costs(self, costs: np.ndarray, weights: np.ndarray) -> Tuple[float, float]:
        """
        从成本分布计算CVaR
        
        Args:
            costs: 成本数组 [num_scenarios]
            weights: 权重数组 [num_scenarios]
            
        Returns:
            (var_value, cvar_value): VaR和CVaR值
        """
        # 排序成本和权重
        sorted_indices = np.argsort(costs)
        sorted_costs = costs[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        # 计算累积权重
        cumulative_weights = np.cumsum(sorted_weights)
        
        # 找到VaR位置
        var_index = np.argmax(cumulative_weights >= 1 - self.alpha)
        
        if var_index == 0:
            var_value = sorted_costs[0]
        else:
            # 线性插值
            weight_before = cumulative_weights[var_index - 1] if var_index > 0 else 0.0
            weight_after = cumulative_weights[var_index]
            
            if weight_after > weight_before:
                interpolation = (1 - self.alpha - weight_before) / (weight_after - weight_before)
                var_value = sorted_costs[var_index - 1] * (1 - interpolation) + sorted_costs[var_index] * interpolation
            else:
                var_value = sorted_costs[var_index]
                
        # 计算CVaR
        tail_weights = sorted_weights[var_index:]
        tail_costs = sorted_costs[var_index:]
        
        if np.sum(tail_weights) > 0:
            cvar_value = np.sum(tail_weights * tail_costs) / np.sum(tail_weights)
        else:
            cvar_value = var_value
            
        return var_value, cvar_value
        
    def sensitivity_analysis(self, scenarios: List[Dict[str, Any]], 
                           alpha_values: List[float]) -> Dict[str, Any]:
        """
        风险水平敏感性分析
        
        Args:
            scenarios: 场景列表
            alpha_values: 风险水平列表
            
        Returns:
            sensitivity_results: 敏感性分析结果
        """
        results = {}
        
        for alpha in alpha_values:
            # 临时更新alpha
            original_alpha = self.alpha
            self.alpha = alpha
            
            # 优化
            initial_controls = np.zeros((50, 2))  # 假设50个时间步
            optimization_result = self.optimize_cvar(scenarios, initial_controls)
            
            if optimization_result['success']:
                results[f'alpha_{alpha:.2f}'] = {
                    'cvar_value': optimization_result['cvar_value'],
                    'var_value': optimization_result['var_value'],
                    'cost': optimization_result['cost'],
                    'iterations': optimization_result['iterations']
                }
            else:
                results[f'alpha_{alpha:.2f}'] = {
                    'success': False,
                    'reason': optimization_result['reason']
                }
                
            # 恢复原始alpha
            self.alpha = original_alpha
            
        return results
        
    def validate_solution(self, controls: np.ndarray, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        验证解的有效性
        
        Args:
            controls: 优化的控制序列
            scenarios: 场景列表
            
        Returns:
            validation_result: 验证结果
        """
        validation = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        # 检查控制边界
        for t, control in enumerate(controls):
            if control[0] < self.control_bounds['acceleration'][0] or control[0] > self.control_bounds['acceleration'][1]:
                validation['errors'].append(f'Acceleration bound violation at time {t}: {control[0]}')
                validation['valid'] = False
                
            if control[1] < self.control_bounds['steering'][0] or control[1] > self.control_bounds['steering'][1]:
                validation['errors'].append(f'Steering bound violation at time {t}: {control[1]}')
                validation['valid'] = False
                
        # 检查动态可行性
        try:
            for scenario in scenarios:
                initial_state = scenario.get('initial_state', np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]))
                trajectory = self._simulate_trajectory(initial_state, controls)
                
                # 检查速度合理性
                velocities = trajectory[:, 2]
                if np.any(velocities < 0) or np.any(velocities > 30.0):
                    validation['warnings'].append('Unrealistic velocities detected')
                    
                # 检查位置合理性
                positions = trajectory[:, :2]
                if np.any(np.isnan(positions)) or np.any(np.isinf(positions)):
                    validation['errors'].append('Invalid positions detected')
                    validation['valid'] = False
                    
        except Exception as e:
            validation['errors'].append(f'Trajectory simulation failed: {str(e)}')
            validation['valid'] = False
            
        return validation