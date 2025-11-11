"""
MARC主规划器

实现MARC论文中的多策略和风险感知应急规划主框架。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
import time
import warnings

from ..scenario.policy_scenario_tree import PolicyScenarioTree
from ..scenario.forward_reachable_set import ForwardReachableSet
from ..scenario.branch_point import BranchPointAnalyzer
from ..planning.risk_aware_planning import RiskAwarePlanning
from ..trajectory.trajectory_tree import MARCTrajectoryTree
from ..trajectory.optimizers.ilqr_optimizer import ILQROptimizer
from ..trajectory.optimizers.mpc_optimizer import MPCOptimizer


class MARCPlanner:
    """MARC规划器主类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 初始化组件
        self.scenario_tree_generator = PolicyScenarioTree(config)
        self.forward_reachable_set = ForwardReachableSet(config)
        self.branch_point_analyzer = BranchPointAnalyzer(config)
        self.risk_aware_planning = RiskAwarePlanning(config)
        self.trajectory_tree = MARCTrajectoryTree(config)
        
        # 优化器
        self.ilqr_optimizer = ILQROptimizer(config)
        self.mpc_optimizer = MPCOptimizer(config)
        
        # 规划参数
        self.planning_horizon = config.get('planning_horizon', 50)
        self.dt = config.get('dt', 0.1)
        self.max_planning_time = config.get('max_planning_time', 1.0)
        self.use_replanning = config.get('use_replanning', True)
        self.replanning_interval = config.get('replanning_interval', 10)
        
        # 策略参数
        self.ego_policies = config.get('ego_policies', ['conservative', 'balanced', 'aggressive'])
        self.risk_alpha = config.get('risk_alpha', 0.1)
        self.confidence_threshold = config.get('confidence_threshold', 0.8)
        
        # 状态
        self.current_trajectory = None
        self.current_controls = None
        self.planning_time = 0.0
        self.planning_success = False
        
    def plan(self, initial_state: np.ndarray, target_trajectory: np.ndarray,
             exo_agents: List[Dict[str, Any]] = None,
             obstacles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        执行MARC规划
        
        Args:
            initial_state: 初始状态 [x, y, v, theta, a, delta]
            target_trajectory: 目标轨迹
            exo_agents: 外部智能体列表
            obstacles: 障碍物列表
            
        Returns:
            planning_result: 规划结果
        """
        start_time = time.time()
        
        try:
            # 1. 生成策略条件场景树
            scenario_result = self._generate_policy_scenarios(
                initial_state, target_trajectory, exo_agents
            )
            
            if not scenario_result['success']:
                return {
                    'success': False,
                    'reason': f'Scenario generation failed: {scenario_result["reason"]}',
                    'trajectory': None,
                    'controls': None
                }
                
            scenario_tree = scenario_result['scenario_tree']
            
            # 2. 分析分支点
            branch_points = self._analyze_branch_points(scenario_tree, target_trajectory)
            
            # 3. 构建轨迹树
            trajectory_result = self._build_trajectory_tree(
                initial_state, target_trajectory, scenario_tree, branch_points
            )
            
            if not trajectory_result['success']:
                return {
                    'success': False,
                    'reason': f'Trajectory tree building failed: {trajectory_result["reason"]}',
                    'trajectory': None,
                    'controls': None
                }
                
            # 4. 风险感知规划
            risk_aware_result = self._risk_aware_planning(
                scenario_tree, initial_state, target_trajectory
            )
            
            if not risk_aware_result['success']:
                return {
                    'success': False,
                    'reason': f'Risk-aware planning failed: {risk_aware_result["reason"]}',
                    'trajectory': None,
                    'controls': None
                }
                
            # 5. 轨迹优化
            optimization_result = self._optimize_trajectory(
                initial_state, target_trajectory, risk_aware_result, obstacles
            )
            
            if not optimization_result['success']:
                return {
                    'success': False,
                    'reason': f'Trajectory optimization failed: {optimization_result["reason"]}',
                    'trajectory': None,
                    'controls': None
                }
                
            # 更新状态
            self.current_trajectory = optimization_result['trajectory']
            self.current_controls = optimization_result['controls']
            self.planning_success = True
            self.planning_time = time.time() - start_time
            
            return {
                'success': True,
                'trajectory': optimization_result['trajectory'],
                'controls': optimization_result['controls'],
                'cost': optimization_result['cost'],
                'planning_time': self.planning_time,
                'scenario_tree': scenario_tree,
                'trajectory_tree': trajectory_result['trajectory_tree'],
                'branch_points': branch_points,
                'risk_metrics': risk_aware_result.get('risk_metrics', {})
            }
            
        except Exception as e:
            self.planning_success = False
            self.planning_time = time.time() - start_time
            
            return {
                'success': False,
                'reason': f'Planning failed with exception: {str(e)}',
                'trajectory': None,
                'controls': None,
                'planning_time': self.planning_time
            }
            
    def _generate_policy_scenarios(self, initial_state: np.ndarray, 
                                 target_trajectory: np.ndarray,
                                 exo_agents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        生成策略条件场景
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            exo_agents: 外部智能体
            
        Returns:
            scenario_result: 场景生成结果
        """
        try:
            # 生成策略条件场景
            scenarios = self.scenario_tree_generator.generate_policy_conditioned_scenarios(
                self.ego_policies, target_trajectory, initial_state
            )
            
            # 添加外部智能体信息
            if exo_agents:
                for scenario in scenarios:
                    scenario['exo_agents'] = exo_agents
                    
            # 构建场景树
            scenario_tree = self.scenario_tree_generator.build_scenario_tree(scenarios)
            
            return {
                'success': True,
                'scenario_tree': scenario_tree,
                'scenarios': scenarios
            }
            
        except Exception as e:
            return {
                'success': False,
                'reason': str(e),
                'scenario_tree': None
            }
            
    def _analyze_branch_points(self, scenario_tree, 
                              target_trajectory: np.ndarray) -> List[int]:
        """
        分析分支点
        
        Args:
            scenario_tree: 场景树
            target_trajectory: 目标轨迹
            
        Returns:
            branch_points: 分支点时间步列表
        """
        try:
            # 分析场景分歧
            divergence_points = self.branch_point_analyzer.analyze_divergence(scenario_tree)
            
            # 选择最优分支点
            optimal_branch_points = self.branch_point_analyzer.select_optimal_branch_points(
                divergence_points, target_trajectory
            )
            
            return optimal_branch_points
            
        except Exception as e:
            warnings.warn(f'Branch point analysis failed: {str(e)}')
            # 默认分支点
            return [10, 20, 30, 40]
            
    def _build_trajectory_tree(self, initial_state: np.ndarray,
                              target_trajectory: np.ndarray,
                              scenario_tree, branch_points: List[int]) -> Dict[str, Any]:
        """
        构建轨迹树
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            scenario_tree: 场景树
            branch_points: 分支点
            
        Returns:
            trajectory_result: 轨迹树构建结果
        """
        try:
            # 初始化轨迹树
            root_id = self.trajectory_tree.initialize_tree(initial_state, target_trajectory)
            
            # 提取场景数据
            scenarios = self._extract_scenarios_from_tree(scenario_tree)
            
            # 扩展轨迹树
            expansion_result = self.trajectory_tree.expand_tree(
                scenarios, target_trajectory, branch_points
            )
            
            if not expansion_result['success']:
                return {
                    'success': False,
                    'reason': expansion_result['reason'],
                    'trajectory_tree': None
                }
                
            return {
                'success': True,
                'trajectory_tree': self.trajectory_tree,
                'statistics': self.trajectory_tree.get_tree_statistics()
            }
            
        except Exception as e:
            return {
                'success': False,
                'reason': str(e),
                'trajectory_tree': None
            }
            
    def _risk_aware_planning(self, scenario_tree, initial_state: np.ndarray,
                           target_trajectory: np.ndarray) -> Dict[str, Any]:
        """
        风险感知规划
        
        Args:
            scenario_tree: 场景树
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            
        Returns:
            risk_result: 风险感知规划结果
        """
        try:
            # 执行风险感知应急规划
            result = self.risk_aware_planning.optimize_contingency_plans(
                scenario_tree, initial_state, target_trajectory
            )
            
            if result['success']:
                # 计算风险指标
                risk_metrics = self._compute_risk_metrics(result)
                result['risk_metrics'] = risk_metrics
                
            return result
            
        except Exception as e:
            return {
                'success': False,
                'reason': str(e),
                'risk_metrics': {}
            }
            
    def _optimize_trajectory(self, initial_state: np.ndarray,
                           target_trajectory: np.ndarray,
                           risk_aware_result: Dict[str, Any],
                           obstacles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        轨迹优化
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            risk_aware_result: 风险感知规划结果
            obstacles: 障碍物
            
        Returns:
            optimization_result: 优化结果
        """
        try:
            # 选择优化器
            optimizer_type = self.config.get('optimizer_type', 'ilqr')
            
            if optimizer_type == 'ilqr':
                # iLQR优化
                initial_controls = risk_aware_result.get('controls', np.zeros((len(target_trajectory), 2)))
                
                result = self.ilqr_optimizer.optimize(
                    initial_state, target_trajectory, initial_controls
                )
                
            elif optimizer_type == 'mpc':
                # MPC优化
                # 设置障碍物
                self.mpc_optimizer.clear_obstacles()
                if obstacles:
                    for obstacle in obstacles:
                        self.mpc_optimizer.add_obstacle(obstacle)
                        
                result = self.mpc_optimizer.optimize(
                    initial_state, target_trajectory, obstacles
                )
                
            else:
                # 默认使用风险感知规划的结果
                result = {
                    'success': True,
                    'trajectory': self._simulate_trajectory(
                        initial_state, risk_aware_result['controls']
                    ),
                    'controls': risk_aware_result['controls'],
                    'cost': risk_aware_result.get('cost', 0.0)
                }
                
            return result
            
        except Exception as e:
            return {
                'success': False,
                'reason': str(e),
                'trajectory': None,
                'controls': None
            }
            
    def _extract_scenarios_from_tree(self, scenario_tree) -> List[Dict[str, Any]]:
        """从场景树提取场景数据"""
        scenarios = []
        
        # 简化处理：创建示例场景
        for i in range(3):
            scenario = {
                'id': f'scenario_{i}',
                'probability': 1.0 / 3.0,
                'ego_trajectory': self._generate_sample_trajectory(50),
                'exo_trajectories': self._generate_sample_exo_trajectories(50, 2),
                'risk_costs': self._compute_sample_risk_costs(50, 2)
            }
            scenarios.append(scenario)
            
        return scenarios
        
    def _generate_sample_trajectory(self, horizon: int) -> np.ndarray:
        """生成示例轨迹"""
        trajectory = np.zeros((horizon, 6))
        
        for t in range(horizon):
            trajectory[t] = [
                10.0 * t * 0.1,  # x
                0.0,              # y
                10.0,             # v
                0.0,              # theta
                0.0,              # a
                0.0               # delta
            ]
            
        return trajectory
        
    def _generate_sample_exo_trajectories(self, horizon: int, num_agents: int) -> List[np.ndarray]:
        """生成示例外部智能体轨迹"""
        trajectories = []
        
        for i in range(num_agents):
            trajectory = np.zeros((horizon, 6))
            
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
        
    def _compute_sample_risk_costs(self, horizon: int, num_agents: int) -> np.ndarray:
        """计算示例风险成本"""
        risk_costs = np.zeros((horizon, num_agents))
        
        for t in range(horizon):
            for i in range(num_agents):
                distance = 5.0 - 0.1 * t
                risk_costs[t, i] = 1000.0 * np.exp(-distance)
                
        return risk_costs
        
    def _simulate_trajectory(self, initial_state: np.ndarray, 
                           controls: np.ndarray) -> np.ndarray:
        """模拟轨迹"""
        trajectory = np.zeros((len(controls) + 1, 6))
        trajectory[0] = initial_state
        
        state = initial_state.copy()
        
        for t, control in enumerate(controls):
            # 自行车模型
            x, y, v, theta, a, delta = state
            da, ddelta = control
            
            a_next = np.clip(a + da * self.dt, -3.0, 3.0)
            delta_next = np.clip(delta + ddelta * self.dt, -0.5, 0.5)
            
            v_next = v + a_next * self.dt
            v_next = max(0.0, v_next)
            
            x_next = x + v_next * np.cos(theta) * self.dt
            y_next = y + v_next * np.sin(theta) * self.dt
            theta_next = theta + v_next / 2.5 * np.tan(delta_next) * self.dt
            
            trajectory[t + 1] = [x_next, y_next, v_next, theta_next, a_next, delta_next]
            state = trajectory[t + 1]
            
        return trajectory
        
    def _compute_risk_metrics(self, risk_aware_result: Dict[str, Any]) -> Dict[str, Any]:
        """计算风险指标"""
        risk_metrics = {}
        
        # CVaR值
        if 'cvar_value' in risk_aware_result:
            risk_metrics['cvar'] = risk_aware_result['cvar_value']
            
        # 风险水平
        risk_metrics['risk_alpha'] = self.risk_alpha
        
        # 规划成功率
        risk_metrics['planning_success_rate'] = 1.0 if risk_aware_result['success'] else 0.0
        
        return risk_metrics
        
    def replan(self, current_state: np.ndarray, target_trajectory: np.ndarray,
              exo_agents: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        重新规划
        
        Args:
            current_state: 当前状态
            target_trajectory: 目标轨迹
            exo_agents: 外部智能体
            
        Returns:
            replanning_result: 重新规划结果
        """
        if not self.use_replanning:
            return {
                'success': False,
                'reason': 'Replanning is disabled',
                'trajectory': self.current_trajectory,
                'controls': self.current_controls
            }
            
        # 执行重新规划
        return self.plan(current_state, target_trajectory, exo_agents)
        
    def get_planning_statistics(self) -> Dict[str, Any]:
        """获取规划统计信息"""
        return {
            'planning_success': self.planning_success,
            'planning_time': self.planning_time,
            'current_trajectory_length': len(self.current_trajectory) if self.current_trajectory is not None else 0,
            'trajectory_tree_stats': self.trajectory_tree.get_tree_statistics() if self.trajectory_tree else {}
        }
        
    def reset(self) -> None:
        """重置规划器状态"""
        self.current_trajectory = None
        self.current_controls = None
        self.planning_success = False
        self.planning_time = 0.0
        
        # 重置轨迹树
        self.trajectory_tree = MARCTrajectoryTree(self.config)