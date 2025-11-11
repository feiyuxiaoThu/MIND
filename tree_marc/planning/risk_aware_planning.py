"""
风险感知应急规划

实现MARC论文中的风险感知应急规划(RCP)算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy.optimize import minimize
from .cvar_optimizer import CVAROptimizer
from .bilevel_optimization import BilevelOptimization


class RiskAwarePlanning:
    """风险感知应急规划"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # RCP参数
        self.alpha = config.get('alpha', 0.1)  # CVaR风险水平
        self.max_iterations = config.get('max_iterations', 50)
        self.tolerance = config.get('tolerance', 1e-6)
        self.risk_weights = config.get('risk_weights', {
            'collision': 1000.0,
            'comfort': 10.0,
            'efficiency': 1.0
        })
        
        # 初始化优化器
        self.cvar_optimizer = CVAROptimizer(config)
        self.bilevel_optimization = BilevelOptimization(config)
        
    def optimize_contingency_plans(self, scenario_tree, initial_state: np.ndarray,
                                  target_trajectory: np.ndarray) -> Dict[str, Any]:
        """
        优化应急规划
        
        Args:
            scenario_tree: 场景树
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            
        Returns:
            optimization_result: 优化结果
        """
        # 提取场景数据
        scenarios = self._extract_scenarios(scenario_tree)
        
        if not scenarios:
            return {
                'success': False,
                'reason': 'No scenarios available',
                'trajectory_tree': None
            }
            
        # 初始化决策变量
        horizon = len(target_trajectory)
        num_scenarios = len(scenarios)
        
        # 初始化控制序列
        initial_controls = np.zeros((horizon, 2))
        
        # 双级优化
        result = self.bilevel_optimization.optimize(
            scenarios, initial_state, target_trajectory, 
            initial_controls, self.cvar_optimizer
        )
        
        return result
        
    def _extract_scenarios(self, scenario_tree) -> List[Dict[str, Any]]:
        """提取场景数据"""
        scenarios = []
        
        # 这里需要从场景树中提取数据
        # 简化处理：创建示例场景
        for i in range(3):  # 创建3个场景
            scenario = {
                'id': f'scenario_{i}',
                'probability': 1.0 / 3.0,
                'ego_trajectory': self._generate_sample_trajectory(50),
                'exo_trajectories': self._generate_sample_exo_trajectories(50, 2),
                'risk_costs': self._compute_risk_costs(50, 2),
                'initial_state': np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                'target_trajectory': self._generate_sample_trajectory(50)
            }
            scenarios.append(scenario)
            
        return scenarios
        
    def _generate_sample_trajectory(self, horizon: int) -> np.ndarray:
        """生成示例轨迹"""
        trajectory = np.zeros((horizon, 6))
        
        # 简单的直行轨迹
        for t in range(horizon):
            trajectory[t] = [
                10.0 * t * 0.1,  # x
                0.0,              # y
                10.0,             # v
                0.0,               # theta
                0.0,               # a
                0.0                # delta
            ]
            
        return trajectory
        
    def _generate_sample_exo_trajectories(self, horizon: int, num_agents: int) -> List[np.ndarray]:
        """生成示例外部智能体轨迹"""
        trajectories = []
        
        for i in range(num_agents):
            trajectory = np.zeros((horizon, 6))
            
            # 简化的直线轨迹，带有偏移
            offset_angle = 2 * np.pi * i / num_agents
            offset_distance = 8.0
            
            for t in range(horizon):
                trajectory[t] = [
                    10.0 * t * 0.1 + offset_distance * np.cos(offset_angle),
                    offset_distance * np.sin(offset_angle),
                    8.0,
                    0.0,
                    0.0,
                    0.0
                ]
                
            trajectories.append(trajectory)
            
        return trajectories
        
    def _compute_risk_costs(self, horizon: int, num_agents: int) -> np.ndarray:
        """计算风险成本"""
        risk_costs = np.zeros((horizon, num_agents))
        
        for t in range(horizon):
            for i in range(num_agents):
                # 简化的风险成本模型
                distance = 5.0 - 0.1 * t  # 随时间接近
                risk_costs[t, i] = self.risk_weights['collision'] * np.exp(-distance)
                
        return risk_costs
        
    def compute_cvar(self, costs: np.ndarray, probabilities: np.ndarray, 
                     alpha: float) -> float:
        """
        计算条件风险价值(CVaR)
        
        Args:
            costs: 风险成本 [num_scenarios]
            probabilities: 概率权重 [num_scenarios]
            alpha: 风险水平 (0-1)
            
        Returns:
            cvar_value: CVaR值
        """
        # 排序成本和概率
        sorted_indices = np.argsort(costs)
        sorted_costs = costs[sorted_indices]
        sorted_probabilities = probabilities[sorted_indices]
        
        # 计算CVaR
        cvar_value = 0.0
        cumulative_prob = 0.0
        
        for i, (cost, prob) in enumerate(zip(sorted_costs, sorted_probabilities)):
            cumulative_prob += prob
            
            if cumulative_prob > (1 - alpha):
                cvar_value = cvar_value + prob * cost
            else:
                cvar_value = cvar_value + prob * cost
                
        return cvar_value
        
    def evaluate_risk_tolerance_sensitivity(self, scenarios: List[Dict[str, Any]], 
                                         alpha_values: List[float]) -> Dict[str, Any]:
        """
        评估风险容忍度敏感性
        
        Args:
            scenarios: 场景列表
            alpha_values: 风险水平列表
            
        Returns:
            sensitivity_analysis: 敏感性分析结果
        """
        sensitivity_results = {}
        
        for alpha in alpha_values:
            # 计算每个场景的CVaR
            cvar_values = []
            for scenario in scenarios:
                cvar = self.compute_cvar(
                    scenario['risk_costs'].flatten(),
                    np.array([scenario['probability']]),
                    alpha
                )
                cvar_values.append(cvar)
                
            sensitivity_results[f'alpha_{alpha:.2f}'] = {
                'mean_cvar': np.mean(cvar_values),
                'std_cvar': np.std(cvar_values),
                'min_cvar': np.min(cvar_values),
                'max_cvar': np.max(cvar_values)
            }
            
        return sensitivity_results
        
    def generate_risk_weighted_trajectories(self, scenarios: List[Dict[str, Any]], 
                                         alpha: float) -> List[np.ndarray]:
        """
        生成风险加权轨迹
        
        Args:
            scenarios: 场景列表
            alpha: 风险水平
            
        Returns:
            weighted_trajectories: 风险加权轨迹列表
        """
        weighted_trajectories = []
        
        for scenario in scenarios:
            # 计算风险权重
            risk_costs = scenario['risk_costs']
            cvar_value = self.compute_cvar(
                risk_costs.flatten(),
                np.array([scenario['probability']]),
                alpha
            )
            
            # 风险权重
            risk_weight = 1.0 / (1.0 + cvar_value)
            
            # 加权轨迹
            weighted_trajectory = scenario['ego_trajectory'].copy()
            weighted_trajectory *= risk_weight
            
            weighted_trajectories.append(weighted_trajectory)
            
        return weighted_trajectories
        
    def optimize_single_scenario(self, scenario: Dict[str, Any], 
                                 initial_state: np.ndarray,
                                 target_trajectory: np.ndarray,
                                 alpha: float) -> Dict[str, Any]:
        """
        优化单个场景的应急规划
        
        Args:
            scenario: 单个场景数据
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            alpha: 风险水平
            
        Returns:
            optimization_result: 优化结果
        """
        horizon = len(target_trajectory)
        
        # 初始化控制变量
        initial_controls = np.zeros((horizon, 2))
        
        # 定义目标函数
        def objective_function(controls_flat):
            controls = controls_flat.reshape(-1, 2)
            
            # 模拟轨迹
            trajectory = self._simulate_trajectory(initial_state, controls)
            
            # 计算基础成本
            base_cost = self._compute_base_cost(trajectory, target_trajectory)
            
            # 计算风险成本
            risk_cost = self._compute_risk_cost(
                trajectory, scenario['risk_costs']
            )
            
            # 计算CVaR
            cvar_value = self.compute_cvar(
                risk_cost.flatten(),
                np.array([scenario['probability']]),
                alpha
            )
            
            # 总成本 = 基础成本 + 风险成本
            total_cost = base_cost + cvar_value
            
            return total_cost
            
        # 优化
        result = minimize(
            objective_function,
            initial_controls.flatten(),
            method='L-BFGS-B',
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if not result.success:
            return {
                'success': False,
                'reason': result.message,
                'controls': initial_controls,
                'trajectory': self._simulate_trajectory(initial_state, initial_controls)
            }
            
        optimized_controls = result.x.reshape(-1, 2)
        optimized_trajectory = self._simulate_trajectory(initial_state, optimized_controls)
        
        return {
            'success': True,
            'controls': optimized_controls,
            'trajectory': optimized_trajectory,
            'cost': result.fun,
            'iterations': result.nit,
            'cvar_value': self.compute_cvar(
                self._compute_risk_cost(optimized_trajectory, scenario['risk_costs']).flatten(),
                np.array([scenario['probability']]),
                alpha
            )
        }
        
    def _simulate_trajectory(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """模拟轨迹"""
        trajectory = np.zeros((len(controls) + 1, 6))
        trajectory[0] = initial_state
        
        state = initial_state.copy()
        
        for t, control in enumerate(controls):
            # 简化的自行车模型
            x, y, v, theta, a, delta = state
            da, ddelta = control
            
            # 更新状态
            a_next = a + da * 0.1
            delta_next = delta + ddelta * 0.1
            
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
            # 位置偏差成本
            position_error = np.linalg.norm(trajectory[t, :2] - target_trajectory[t, :2])
            cost += position_error * self.risk_weights['efficiency']
            
            # 速度偏差成本
            velocity_error = abs(trajectory[t, 2] - target_trajectory[t, 2])
            cost += velocity_error * self.risk_weights['efficiency']
            
        return cost
        
    def _compute_risk_cost(self, trajectory: np.ndarray, risk_costs: np.ndarray) -> np.ndarray:
        """计算风险成本"""
        # 确保长度匹配
        min_length = min(len(trajectory), risk_costs.shape[0])
        
        if risk_costs.shape[1] == 0:
            return np.zeros(min_length)
            
        total_risk_cost = np.zeros(min_length)
        
        for t in range(min_length):
            # 累加所有智能体的风险成本
            total_risk_cost[t] = np.sum(risk_costs[t, :])
            
        return total_risk_cost
        
    def analyze_planning_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """分析规划结果"""
        if not results:
            return {
                'analysis': 'No results to analyze',
                'statistics': {}
            }
            
        # 统计信息
        costs = [r['cost'] for r in results if r['success']]
        cvar_values = [r.get('cvar_value', 0.0) for r in results if r['success']]
        
        analysis = {
            'num_successful': len([r for r in results if r['success']]),
            'num_failed': len([r for r in results if not r['success']]),
            'mean_cost': np.mean(costs) if costs else 0.0,
            'std_cost': np.std(costs) if costs else 0.0,
            'mean_cvar': np.mean(cvar_values) if cvar_values else 0.0,
            'std_cvar': np.std(cvar_values) if cvar_values else 0.0,
            'min_cost': np.min(costs) if costs else 0.0,
            'max_cost': np.max(costs) if costs else 0.0,
            'min_cvar': np.min(cvar_values) if cvar_values else 0.0,
            'max_cvar': np.max(cvar_values) if cvar_values else 0.0
        }
        
        # 失败原因分析
        failed_reasons = [r['reason'] for r in results if not r['success']]
        if failed_reasons:
            analysis['failure_reasons'] = failed_reasons
            
        return analysis