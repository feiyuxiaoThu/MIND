"""
双级优化模块

实现MARC论文中的双级优化算法，结合线性规划和iLQR优化。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional, Callable
from scipy.optimize import minimize, linprog
from .cvar_optimizer import CVAROptimizer
import warnings


class BilevelOptimization:
    """双级优化器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 双级优化参数
        self.max_outer_iterations = config.get('max_outer_iterations', 20)
        self.max_inner_iterations = config.get('max_inner_iterations', 50)
        self.outer_tolerance = config.get('outer_tolerance', 1e-4)
        self.inner_tolerance = config.get('inner_tolerance', 1e-6)
        
        # 线性规划参数
        self.lp_method = config.get('lp_method', 'highs')
        self.lp_tolerance = config.get('lp_tolerance', 1e-8)
        
        # iLQR参数
        self.ilqr_max_iterations = config.get('ilqr_max_iterations', 100)
        self.ilqr_tolerance = config.get('ilqr_tolerance', 1e-6)
        self.ilqr_lambda_init = config.get('ilqr_lambda_init', 1.0)
        self.ilqr_lambda_factor = config.get('ilqr_lambda_factor', 10.0)
        
        # CVaR优化器
        self.cvar_optimizer = CVAROptimizer(config)
        
    def optimize(self, scenarios: List[Dict[str, Any]], initial_state: np.ndarray,
                target_trajectory: np.ndarray, initial_controls: np.ndarray,
                cvar_optimizer: CVAROptimizer) -> Dict[str, Any]:
        """
        执行双级优化
        
        Args:
            scenarios: 场景列表
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            initial_controls: 初始控制序列
            cvar_optimizer: CVaR优化器
            
        Returns:
            optimization_result: 优化结果
        """
        if not scenarios:
            return {
                'success': False,
                'reason': 'No scenarios provided',
                'trajectory_tree': None
            }
            
        # 初始化
        horizon = len(target_trajectory)
        num_scenarios = len(scenarios)
        
        # 外层循环：策略优化
        best_controls = initial_controls.copy()
        best_cost = float('inf')
        best_trajectory_tree = None
        
        for outer_iter in range(self.max_outer_iterations):
            # 上层问题：线性规划确定策略权重
            policy_result = self._solve_upper_level_problem(
                scenarios, best_controls, target_trajectory
            )
            
            if not policy_result['success']:
                continue
                
            policy_weights = policy_result['policy_weights']
            
            # 下层问题：iLQR优化控制序列
            control_result = self._solve_lower_level_problem(
                scenarios, initial_state, target_trajectory, 
                best_controls, policy_weights
            )
            
            if not control_result['success']:
                continue
                
            updated_controls = control_result['controls']
            updated_cost = control_result['cost']
            
            # 收敛检查
            if abs(updated_cost - best_cost) < self.outer_tolerance:
                break
                
            # 更新最优解
            if updated_cost < best_cost:
                best_controls = updated_controls
                best_cost = updated_cost
                best_trajectory_tree = control_result['trajectory_tree']
                
        # 构建轨迹树
        if best_trajectory_tree is None:
            best_trajectory_tree = self._build_trajectory_tree(
                scenarios, best_controls, initial_state
            )
            
        return {
            'success': True,
            'controls': best_controls,
            'trajectory_tree': best_trajectory_tree,
            'cost': best_cost,
            'outer_iterations': outer_iter + 1
        }
        
    def _solve_upper_level_problem(self, scenarios: List[Dict[str, Any]], 
                                  controls: np.ndarray, 
                                  target_trajectory: np.ndarray) -> Dict[str, Any]:
        """
        求解上层问题：线性规划确定策略权重
        
        Args:
            scenarios: 场景列表
            controls: 当前控制序列
            target_trajectory: 目标轨迹
            
        Returns:
            policy_result: 策略优化结果
        """
        num_scenarios = len(scenarios)
        num_policies = 3  # 假设3种策略：保守、平衡、激进
        
        # 计算每个场景在不同策略下的成本
        scenario_costs = np.zeros((num_scenarios, num_policies))
        
        for i, scenario in enumerate(scenarios):
            for j, policy in enumerate(['conservative', 'balanced', 'aggressive']):
                # 调整控制策略
                policy_controls = self._adjust_controls_for_policy(controls, policy)
                
                # 计算场景成本
                cost = self._compute_scenario_cost(policy_controls, scenario, target_trajectory)
                scenario_costs[i, j] = cost
                
        # 线性规划：最小化加权成本
        # 目标函数：min sum(w_i * cost_i)
        c = np.mean(scenario_costs, axis=0)  # 平均成本作为目标系数
        
        # 约束条件
        # 1. 权重和为1
        A_eq = np.ones((1, num_policies))
        b_eq = np.array([1.0])
        
        # 2. 权重非负
        bounds = [(0.0, 1.0) for _ in range(num_policies)]
        
        # 3. 风险约束：CVaR不超过阈值
        # 这里简化处理，实际应该包含CVaR约束
        A_ub = None
        b_ub = None
        
        try:
            result = linprog(
                c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                bounds=bounds, method=self.lp_method, tol=self.lp_tolerance
            )
            
            if not result.success:
                return {
                    'success': False,
                    'reason': result.message,
                    'policy_weights': np.ones(num_policies) / num_policies
                }
                
            policy_weights = result.x
            
            return {
                'success': True,
                'policy_weights': policy_weights,
                'scenario_costs': scenario_costs,
                'objective_value': result.fun
            }
            
        except Exception as e:
            return {
                'success': False,
                'reason': f'Linear programming failed: {str(e)}',
                'policy_weights': np.ones(num_policies) / num_policies
            }
            
    def _solve_lower_level_problem(self, scenarios: List[Dict[str, Any]], 
                                  initial_state: np.ndarray,
                                  target_trajectory: np.ndarray,
                                  initial_controls: np.ndarray,
                                  policy_weights: np.ndarray) -> Dict[str, Any]:
        """
        求解下层问题：iLQR优化控制序列
        
        Args:
            scenarios: 场景列表
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            initial_controls: 初始控制序列
            policy_weights: 策略权重
            
        Returns:
            control_result: 控制优化结果
        """
        # 使用iLQR优化控制序列
        ilqr_result = self._ilqr_optimization(
            scenarios, initial_state, target_trajectory, 
            initial_controls, policy_weights
        )
        
        if not ilqr_result['success']:
            return {
                'success': False,
                'reason': 'iLQR optimization failed',
                'controls': initial_controls
            }
            
        # 构建轨迹树
        trajectory_tree = self._build_trajectory_tree(
            scenarios, ilqr_result['controls'], initial_state
        )
        
        return {
            'success': True,
            'controls': ilqr_result['controls'],
            'trajectory_tree': trajectory_tree,
            'cost': ilqr_result['cost'],
            'iterations': ilqr_result['iterations']
        }
        
    def _adjust_controls_for_policy(self, controls: np.ndarray, policy: str) -> np.ndarray:
        """
        根据策略调整控制序列
        
        Args:
            controls: 原始控制序列
            policy: 策略类型 ('conservative', 'balanced', 'aggressive')
            
        Returns:
            adjusted_controls: 调整后的控制序列
        """
        adjusted_controls = controls.copy()
        
        if policy == 'conservative':
            # 保守策略：减小加速度和转向角
            adjusted_controls[:, 0] *= 0.7  # 减小加速度
            adjusted_controls[:, 1] *= 0.6  # 减小转向角
            
        elif policy == 'balanced':
            # 平衡策略：保持原始控制
            pass
            
        elif policy == 'aggressive':
            # 激进策略：增大加速度和转向角
            adjusted_controls[:, 0] *= 1.3  # 增大加速度
            adjusted_controls[:, 1] *= 1.2  # 增大转向角
            
        return adjusted_controls
        
    def _compute_scenario_cost(self, controls: np.ndarray, scenario: Dict[str, Any], 
                              target_trajectory: np.ndarray) -> float:
        """
        计算场景成本
        
        Args:
            controls: 控制序列
            scenario: 场景数据
            target_trajectory: 目标轨迹
            
        Returns:
            scenario_cost: 场景成本
        """
        # 模拟轨迹
        initial_state = scenario.get('initial_state', np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]))
        trajectory = self._simulate_trajectory(initial_state, controls)
        
        # 计算基础成本
        base_cost = self._compute_base_cost(trajectory, target_trajectory)
        
        # 计算风险成本
        risk_cost = self._compute_risk_cost(trajectory, scenario)
        
        return base_cost + risk_cost
        
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
        
    def _compute_base_cost(self, trajectory: np.ndarray, target_trajectory: np.ndarray) -> float:
        """计算基础成本"""
        cost = 0.0
        
        # 确保长度匹配
        min_length = min(len(trajectory), len(target_trajectory))
        
        for t in range(min_length):
            # 位置偏差
            position_error = np.linalg.norm(trajectory[t, :2] - target_trajectory[t, :2])
            cost += position_error
            
            # 速度偏差
            velocity_error = abs(trajectory[t, 2] - target_trajectory[t, 2])
            cost += velocity_error * 0.1
            
        return cost
        
    def _compute_risk_cost(self, trajectory: np.ndarray, scenario: Dict[str, Any]) -> float:
        """计算风险成本"""
        exo_trajectories = scenario.get('exo_trajectories', [])
        
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
                        
        return total_risk_cost
        
    def _ilqr_optimization(self, scenarios: List[Dict[str, Any]], initial_state: np.ndarray,
                          target_trajectory: np.ndarray, initial_controls: np.ndarray,
                          policy_weights: np.ndarray) -> Dict[str, Any]:
        """
        iLQR优化算法（简化版本）
        
        Args:
            scenarios: 场景列表
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            initial_controls: 初始控制序列
            policy_weights: 策略权重
            
        Returns:
            ilqr_result: iLQR优化结果
        """
        controls = initial_controls.copy()
        
        # 简化的iLQR：直接返回初始控制
        cost = 0.0
        for scenario in scenarios:
            scenario_cost = self._compute_scenario_cost(controls, scenario, target_trajectory)
            cost += scenario_cost
            
        return {
            'success': True,
            'controls': controls,
            'cost': cost,
            'iterations': 1
        }
        
    def _build_trajectory_tree(self, scenarios: List[Dict[str, Any]], 
                              controls: np.ndarray, initial_state: np.ndarray) -> Dict[str, Any]:
        """
        构建轨迹树（简化版本）
        
        Args:
            scenarios: 场景列表
            controls: 优化后的控制序列
            initial_state: 初始状态
            
        Returns:
            trajectory_tree: 轨迹树结构
        """
        trajectory_tree = {
            'root': {
                'state': initial_state,
                'controls': controls,
                'children': []
            },
            'scenarios': []
        }
        
        # 为每个场景生成轨迹
        for i, scenario in enumerate(scenarios):
            trajectory = self._simulate_trajectory(initial_state, controls)
            
            scenario_node = {
                'id': f'scenario_{i}',
                'probability': scenario.get('probability', 1.0 / len(scenarios)),
                'trajectory': trajectory,
                'cost': self._compute_scenario_cost(controls, scenario, 
                                                   scenario.get('target_trajectory', trajectory))
            }
            
            trajectory_tree['scenarios'].append(scenario_node)
            
        return trajectory_tree