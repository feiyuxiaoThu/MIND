"""
MARC规划器测试

测试MARC规划器的各个组件功能。
"""

import unittest
import numpy as np
import json
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree_marc.planners.mind_planner import MARCPlanner
from tree_marc.planning.cvar_optimizer import CVAROptimizer
from tree_marc.planning.bilevel_optimization import BilevelOptimization
from tree_marc.trajectory.trajectory_tree import MARCTrajectoryTree


class TestMARCPlanner(unittest.TestCase):
    """MARC规划器测试类"""
    
    def setUp(self):
        """测试设置"""
        # 加载配置
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'marc_config.json')
        with open(config_path, 'r') as f:
            self.config = json.load(f)
            
        # 创建规划器
        self.planner = MARCPlanner(self.config)
        
        # 测试数据
        self.initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.target_trajectory = self._generate_target_trajectory(30)
        
    def _generate_target_trajectory(self, horizon: int) -> np.ndarray:
        """生成目标轨迹"""
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
        
    def test_planner_initialization(self):
        """测试规划器初始化"""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.planning_horizon, 50)
        self.assertEqual(self.planner.dt, 0.1)
        self.assertTrue(self.planner.use_replanning)
        
    def test_basic_planning(self):
        """测试基本规划功能"""
        result = self.planner.plan(self.initial_state, self.target_trajectory)
        
        # 检查结果结构
        self.assertIn('success', result)
        self.assertIn('trajectory', result)
        self.assertIn('controls', result)
        self.assertIn('planning_time', result)
        
        if result['success']:
            self.assertIsNotNone(result['trajectory'])
            self.assertIsNotNone(result['controls'])
            self.assertGreater(result['planning_time'], 0)
            
            # 检查轨迹维度
            expected_length = len(self.target_trajectory) + 1
            self.assertEqual(result['trajectory'].shape[0], expected_length)
            self.assertEqual(result['trajectory'].shape[1], 6)
            
            # 检查控制序列维度
            self.assertEqual(result['controls'].shape[0], len(self.target_trajectory))
            self.assertEqual(result['controls'].shape[1], 2)
            
    def test_planning_with_obstacles(self):
        """测试带障碍物的规划"""
        obstacles = [
            {'x': 15.0, 'y': 2.0, 'radius': 2.0},
            {'x': 25.0, 'y': -1.0, 'radius': 1.5}
        ]
        
        result = self.planner.plan(self.initial_state, self.target_trajectory, 
                                 obstacles=obstacles)
        
        self.assertIn('success', result)
        
    def test_planning_with_exo_agents(self):
        """测试带外部智能体的规划"""
        exo_agents = [
            {
                'id': 'vehicle_1',
                'initial_state': np.array([20.0, 0.0, 8.0, 0.0, 0.0, 0.0]),
                'trajectory': np.array([
                    [20.0 + 8.0 * t * 0.1, 0.0, 8.0, 0.0, 0.0, 0.0] 
                    for t in range(30)
                ])
            }
        ]
        
        result = self.planner.plan(self.initial_state, self.target_trajectory, 
                                 exo_agents=exo_agents)
        
        self.assertIn('success', result)
        
    def test_replanning(self):
        """测试重新规划功能"""
        # 首次规划
        result = self.planner.plan(self.initial_state, self.target_trajectory)
        
        if result['success']:
            # 重新规划
            current_state = result['trajectory'][5]
            replan_result = self.planner.replan(current_state, self.target_trajectory)
            
            self.assertIn('success', replan_result)
            
    def test_planning_statistics(self):
        """测试规划统计信息"""
        result = self.planner.plan(self.initial_state, self.target_trajectory)
        
        stats = self.planner.get_planning_statistics()
        
        self.assertIn('planning_success', stats)
        self.assertIn('planning_time', stats)
        self.assertIn('current_trajectory_length', stats)
        
    def test_planner_reset(self):
        """测试规划器重置"""
        result = self.planner.plan(self.initial_state, self.target_trajectory)
        
        # 重置规划器
        self.planner.reset()
        
        # 检查状态
        stats = self.planner.get_planning_statistics()
        self.assertIsNone(self.planner.current_trajectory)
        self.assertIsNone(self.planner.current_controls)
        self.assertFalse(self.planner.planning_success)


class TestCVAROptimizer(unittest.TestCase):
    """CVaR优化器测试类"""
    
    def setUp(self):
        """测试设置"""
        config = {
            'alpha': 0.1,
            'max_iterations': 50,
            'tolerance': 1e-6,
            'control_bounds': {
                'acceleration': [-3.0, 3.0],
                'steering': [-0.5, 0.5]
            }
        }
        
        self.optimizer = CVAROptimizer(config)
        
        # 测试场景
        self.scenarios = [
            {
                'id': 'scenario_1',
                'probability': 0.5,
                'initial_state': np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                'target_trajectory': np.array([
                    [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
                    for t in range(20)
                ])
            },
            {
                'id': 'scenario_2',
                'probability': 0.5,
                'initial_state': np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                'target_trajectory': np.array([
                    [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
                    for t in range(20)
                ])
            }
        ]
        
        self.initial_controls = np.zeros((20, 2))
        
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.alpha, 0.1)
        self.assertEqual(self.optimizer.max_iterations, 50)
        
    def test_cvar_optimization(self):
        """测试CVaR优化"""
        result = self.optimizer.optimize_cvar(self.scenarios, self.initial_controls)
        
        self.assertIn('success', result)
        self.assertIn('controls', result)
        self.assertIn('cvar_value', result)
        
        if result['success']:
            self.assertEqual(result['controls'].shape, self.initial_controls.shape)
            self.assertIsInstance(result['cvar_value'], float)
            
    def test_cvar_computation(self):
        """测试CVaR计算"""
        costs = np.array([10.0, 20.0, 30.0, 40.0])
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        var_value, cvar_value = self.optimizer.compute_cvar_from_costs(costs, weights)
        
        self.assertIsInstance(var_value, float)
        self.assertIsInstance(cvar_value, float)
        self.assertGreaterEqual(cvar_value, var_value)
        
    def test_sensitivity_analysis(self):
        """测试敏感性分析"""
        alpha_values = [0.05, 0.1, 0.2]
        
        result = self.optimizer.sensitivity_analysis(self.scenarios, alpha_values)
        
        self.assertIsInstance(result, dict)
        for alpha in alpha_values:
            key = f'alpha_{alpha:.2f}'
            self.assertIn(key, result)


class TestBilevelOptimization(unittest.TestCase):
    """双级优化测试类"""
    
    def setUp(self):
        """测试设置"""
        config = {
            'max_outer_iterations': 10,
            'max_inner_iterations': 20,
            'outer_tolerance': 1e-4,
            'inner_tolerance': 1e-6
        }
        
        self.optimizer = BilevelOptimization(config)
        
        # 测试数据
        self.scenarios = [
            {
                'id': 'scenario_1',
                'probability': 0.5,
                'initial_state': np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]),
                'target_trajectory': np.array([
                    [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
                    for t in range(20)
                ])
            }
        ]
        
        self.initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.target_trajectory = np.array([
            [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
            for t in range(20)
        ])
        self.initial_controls = np.zeros((20, 2))
        
    def test_optimizer_initialization(self):
        """测试优化器初始化"""
        self.assertIsNotNone(self.optimizer)
        self.assertEqual(self.optimizer.max_outer_iterations, 10)
        self.assertEqual(self.optimizer.max_inner_iterations, 20)
        
    def test_bilevel_optimization(self):
        """测试双级优化"""
        cvar_optimizer = CVAROptimizer({
            'alpha': 0.1,
            'max_iterations': 50,
            'tolerance': 1e-6
        })
        
        result = self.optimizer.optimize(
            self.scenarios, self.initial_state, self.target_trajectory,
            self.initial_controls, cvar_optimizer
        )
        
        self.assertIn('success', result)
        self.assertIn('controls', result)
        self.assertIn('trajectory_tree', result)
        
        if result['success']:
            self.assertEqual(result['controls'].shape, self.initial_controls.shape)


class TestTrajectoryTree(unittest.TestCase):
    """轨迹树测试类"""
    
    def setUp(self):
        """测试设置"""
        config = {
            'max_depth': 10,
            'branching_factor': 3,
            'prune_threshold': 1000.0,
            'min_probability': 0.01
        }
        
        self.tree = MARCTrajectoryTree(config)
        
        self.initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.target_trajectory = np.array([
            [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
            for t in range(20)
        ])
        
    def test_tree_initialization(self):
        """测试树初始化"""
        self.assertIsNotNone(self.tree)
        self.assertEqual(self.tree.max_depth, 10)
        self.assertEqual(self.tree.branching_factor, 3)
        
    def test_tree_initialization_with_trajectory(self):
        """测试轨迹树初始化"""
        root_id = self.tree.initialize_tree(self.initial_state, self.target_trajectory)
        
        self.assertIsNotNone(root_id)
        self.assertEqual(self.tree.graph.number_of_nodes(), 1)
        self.assertEqual(len(self.tree.leaf_nodes), 1)
        
    def test_tree_expansion(self):
        """测试树扩展"""
        root_id = self.tree.initialize_tree(self.initial_state, self.target_trajectory)
        
        scenarios = [
            {
                'id': 'scenario_1',
                'probability': 0.5,
                'target_trajectory': self.target_trajectory
            }
        ]
        
        branch_points = [5, 10, 15]
        
        result = self.tree.expand_tree(scenarios, self.target_trajectory, branch_points)
        
        self.assertIn('success', result)
        self.assertIn('added_nodes', result)
        
    def test_optimal_trajectory_extraction(self):
        """测试最优轨迹提取"""
        root_id = self.tree.initialize_tree(self.initial_state, self.target_trajectory)
        
        states, controls, cost = self.tree.get_optimal_trajectory()
        
        self.assertIsInstance(states, list)
        self.assertIsInstance(controls, list)
        self.assertIsInstance(cost, float)
        
    def test_tree_statistics(self):
        """测试树统计信息"""
        root_id = self.tree.initialize_tree(self.initial_state, self.target_trajectory)
        
        stats = self.tree.get_tree_statistics()
        
        self.assertIn('num_nodes', stats)
        self.assertIn('num_edges', stats)
        self.assertIn('num_leaf_nodes', stats)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)