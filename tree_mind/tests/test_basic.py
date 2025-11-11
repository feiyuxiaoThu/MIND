"""
基础测试模块

测试MIND重构版本的基本功能。
"""

import unittest
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.scenario.scenario_tree import ScenarioTree, ScenarioData, AgentPrediction
from tree.scenario.aime import AIME
from tree.scenario.multimodal import MultimodalProcessor
from tree.trajectory.trajectory_tree import TrajectoryTree, TrajectoryState, ControlInput
from tree.trajectory.dynamics import BicycleDynamics
from tree.trajectory.costs import SafetyCost, TargetCost
from tree.planners.mind_planner import MINDPlanner


class TestScenarioTree(unittest.TestCase):
    """测试场景树"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'max_depth': 5,
            'uncertainty_threshold': 9.0,
            'probability_threshold': 0.001
        }
        self.scenario_tree = ScenarioTree(self.config)
        
    def test_scenario_tree_creation(self):
        """测试场景树创建"""
        self.assertIsNotNone(self.scenario_tree)
        self.assertEqual(self.scenario_tree.max_depth, 5)
        
    def test_add_root(self):
        """测试添加根节点"""
        # 创建测试数据
        means = np.zeros((10, 2))
        covs = np.zeros((10, 2, 2))
        ego_pred = AgentPrediction(means, covs, 1.0)
        scenario_data = ScenarioData(ego_pred, [], 1.0, 0.0, {})
        
        # 添加根节点
        root_node = self.scenario_tree.add_root(scenario_data)
        
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.depth, 0)
        self.assertEqual(self.scenario_tree.get_root().key, root_node.key)
        
    def test_add_scenario(self):
        """测试添加场景"""
        # 添加根节点
        means = np.zeros((10, 2))
        covs = np.zeros((10, 2, 2))
        ego_pred = AgentPrediction(means, covs, 1.0)
        root_data = ScenarioData(ego_pred, [], 1.0, 0.0, {})
        self.scenario_tree.add_root(root_data)
        
        # 添加子场景
        child_data = ScenarioData(ego_pred, [], 0.5, 0.0, {})
        child_node = self.scenario_tree.add_scenario("root", child_data)
        
        self.assertIsNotNone(child_node)
        self.assertEqual(child_node.depth, 1)
        self.assertEqual(child_node.parent_key, "root")
        
    def test_get_statistics(self):
        """测试统计信息"""
        stats = self.scenario_tree.get_statistics()
        
        self.assertIn('total_nodes', stats)
        self.assertIn('max_depth', stats)
        self.assertIn('num_branches', stats)
        self.assertIn('num_end_nodes', stats)


class TestAIME(unittest.TestCase):
    """测试AIME算法"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'max_depth': 3,
            'uncertainty_threshold': 9.0,
            'probability_threshold': 0.001
        }
        self.aime = AIME(self.config)
        
    def test_aime_creation(self):
        """测试AIME创建"""
        self.assertIsNotNone(self.aime)
        self.assertEqual(self.aime.max_depth, 3)
        
    def test_generate_variant_prediction(self):
        """测试生成变体预测"""
        # 创建基础预测
        means = np.random.randn(20, 2)
        covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(20, axis=0)
        base_prediction = AgentPrediction(means, covs, 1.0)
        
        # 生成变体
        variant = self.aime._create_child_scenario(
            ScenarioData(base_prediction, [], 1.0, 0.0, {}), 1
        )
        
        self.assertIsNotNone(variant)
        self.assertEqual(variant.ego_prediction.means.shape, base_prediction.means.shape)
        
    def test_topology_similarity(self):
        """测试拓扑相似性"""
        # 创建两个场景
        means1 = np.random.randn(20, 2)
        covs1 = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(20, axis=0)
        scenario1 = ScenarioData(
            AgentPrediction(means1, covs1, 0.5), [], 0.5, 0.0, {}
        )
        
        means2 = means1 + np.random.randn(20, 2) * 0.01  # 小扰动
        covs2 = covs1
        scenario2 = ScenarioData(
            AgentPrediction(means2, covs2, 0.5), [], 0.5, 0.0, {}
        )
        
        # 测试相似性
        similar = self.aime._are_topologically_similar(scenario1, scenario2)
        self.assertTrue(similar)


class TestTrajectoryTree(unittest.TestCase):
    """测试轨迹树"""
    
    def setUp(self):
        """设置测试"""
        self.config = {'dt': 0.1, 'horizon': 50}
        self.trajectory_tree = TrajectoryTree(self.config)
        
    def test_trajectory_tree_creation(self):
        """测试轨迹树创建"""
        self.assertIsNotNone(self.trajectory_tree)
        self.assertEqual(self.trajectory_tree.dt, 0.1)
        
    def test_add_root(self):
        """测试添加根节点"""
        initial_state = TrajectoryState(
            position=np.array([0.0, 0.0]),
            velocity=10.0,
            heading=0.0,
            acceleration=0.0,
            steering_angle=0.0,
            timestamp=0.0
        )
        initial_control = ControlInput(acceleration=0.0, steering_rate=0.0)
        
        root_node = self.trajectory_tree.add_root(initial_state, initial_control)
        
        self.assertIsNotNone(root_node)
        self.assertEqual(root_node.depth, 0)
        
    def test_add_trajectory_step(self):
        """测试添加轨迹步骤"""
        # 添加根节点
        initial_state = TrajectoryState(
            position=np.array([0.0, 0.0]),
            velocity=10.0,
            heading=0.0,
            acceleration=0.0,
            steering_angle=0.0,
            timestamp=0.0
        )
        initial_control = ControlInput(acceleration=0.0, steering_rate=0.0)
        self.trajectory_tree.add_root(initial_state, initial_control)
        
        # 添加轨迹步骤
        next_state = TrajectoryState(
            position=np.array([1.0, 0.0]),
            velocity=10.0,
            heading=0.0,
            acceleration=0.0,
            steering_angle=0.0,
            timestamp=0.1
        )
        next_control = ControlInput(acceleration=0.0, steering_rate=0.0)
        
        trajectory_node = self.trajectory_tree.add_trajectory_step(
            "root", next_state, next_control, 1.0
        )
        
        self.assertIsNotNone(trajectory_node)
        self.assertEqual(trajectory_node.depth, 1)
        
    def test_compute_trajectory_cost(self):
        """测试计算轨迹成本"""
        # 创建简单轨迹
        initial_state = TrajectoryState(
            position=np.array([0.0, 0.0]),
            velocity=10.0,
            heading=0.0,
            acceleration=0.0,
            steering_angle=0.0,
            timestamp=0.0
        )
        initial_control = ControlInput(acceleration=0.0, steering_rate=0.0)
        self.trajectory_tree.add_root(initial_state, initial_control)
        
        # 添加几个步骤
        parent_id = "root"
        for i in range(3):
            next_state = TrajectoryState(
                position=np.array([float(i+1), 0.0]),
                velocity=10.0,
                heading=0.0,
                acceleration=0.0,
                steering_angle=0.0,
                timestamp=float(i+1) * 0.1
            )
            next_control = ControlInput(acceleration=0.0, steering_rate=0.0)
            trajectory_node = self.trajectory_tree.add_trajectory_step(
                parent_id, next_state, next_control, 1.0
            )
            parent_id = trajectory_node.key
            
        # 获取轨迹并计算成本
        trajectories = self.trajectory_tree.get_all_trajectories()
        if trajectories:
            cost = self.trajectory_tree.compute_trajectory_cost(trajectories[0])
            self.assertGreater(cost, 0)


class TestDynamics(unittest.TestCase):
    """测试动力学模型"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'dt': 0.1,
            'wheelbase': 2.5,
            'max_acceleration': 3.0,
            'max_deceleration': 5.0
        }
        self.dynamics = BicycleDynamics(self.config)
        
    def test_dynamics_creation(self):
        """测试动力学模型创建"""
        self.assertIsNotNone(self.dynamics)
        self.assertEqual(self.dynamics.dt, 0.1)
        self.assertEqual(self.dynamics.wheelbase, 2.5)
        
    def test_state_step(self):
        """测试状态转移"""
        state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # [x, y, v, theta, a, delta]
        control = np.array([0.0, 0.0])  # [da, ddelta]
        
        next_state = self.dynamics.step(state, control)
        
        self.assertEqual(next_state.shape, state.shape)
        self.assertGreater(next_state[0], 0.0)  # x应该增加
        self.assertAlmostEqual(next_state[1], 0.0, places=5)  # y应该保持0
        
    def test_jacobian(self):
        """测试雅可比矩阵"""
        state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0])
        
        A, B = self.dynamics.get_jacobian(state, control)
        
        self.assertEqual(A.shape, (6, 6))
        self.assertEqual(B.shape, (6, 2))
        
    def test_state_validation(self):
        """测试状态验证"""
        # 有效状态
        valid_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        self.assertTrue(self.dynamics.validate_state(valid_state))
        
        # 无效状态（负速度）
        invalid_state = np.array([0.0, 0.0, -1.0, 0.0, 0.0, 0.0])
        self.assertFalse(self.dynamics.validate_state(invalid_state))


class TestCosts(unittest.TestCase):
    """测试成本函数"""
    
    def setUp(self):
        """设置测试"""
        self.safety_config = {'safety_distance': 2.0, 'collision_penalty': 100.0}
        self.target_config = {'target_position': np.array([10.0, 0.0]), 'target_velocity': 10.0}
        
        self.safety_cost = SafetyCost(self.safety_config)
        self.target_cost = TargetCost(self.target_config)
        
    def test_safety_cost(self):
        """测试安全成本"""
        state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0])
        
        # 设置障碍物
        obstacle = np.array([1.0, 0.0])
        self.safety_cost.set_obstacles([obstacle])
        
        cost = self.safety_cost.compute(state, control)
        self.assertGreater(cost, 0.0)
        
    def test_target_cost(self):
        """测试目标成本"""
        state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        control = np.array([0.0, 0.0])
        
        cost = self.target_cost.compute(state, control)
        self.assertGreater(cost, 0.0)
        
        # 更接近目标应该有更低的成本
        closer_state = np.array([5.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        closer_cost = self.target_cost.compute(closer_state, control)
        self.assertLess(closer_cost, cost)


class TestMINDPlanner(unittest.TestCase):
    """测试MIND规划器"""
    
    def setUp(self):
        """设置测试"""
        self.config = {
            'dt': 0.1,
            'horizon': 30,
            'optimizer_type': 'ilqr',
            'target_velocity': 10.0
        }
        self.planner = MINDPlanner(self.config)
        
    def test_planner_creation(self):
        """测试规划器创建"""
        self.assertIsNotNone(self.planner)
        self.assertEqual(self.planner.optimizer_type, 'ilqr')
        
    def test_generate_scenario_data(self):
        """测试生成场景数据"""
        # 创建预测
        means = np.zeros((30, 2))
        covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(30, axis=0)
        predictions = [AgentPrediction(means, covs, 1.0)]
        
        scenario_data_list = self.planner._generate_scenario_data(predictions)
        
        self.assertGreater(len(scenario_data_list), 0)
        
    def test_generate_target_trajectory(self):
        """测试生成目标轨迹"""
        current_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        target_lane = np.array([[x, 0.0] for x in np.linspace(0, 50, 50)])
        
        target_trajectory = self.planner._generate_target_trajectory(
            current_state, target_lane
        )
        
        self.assertEqual(target_trajectory.shape[0], self.config['horizon'])
        self.assertEqual(target_trajectory.shape[1], 6)
        
    def test_switch_optimizer(self):
        """测试切换优化器"""
        original_type = self.planner.optimizer_type
        
        # 切换到MPC
        self.planner.switch_optimizer('mpc')
        self.assertEqual(self.planner.optimizer_type, 'mpc')
        
        # 切换到CBF
        self.planner.switch_optimizer('cbf')
        self.assertEqual(self.planner.optimizer_type, 'cbf')
        
        # 切换回iLQR
        self.planner.switch_optimizer('ilqr')
        self.assertEqual(self.planner.optimizer_type, 'ilqr')


def run_basic_tests():
    """运行基础测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_classes = [
        TestScenarioTree,
        TestAIME,
        TestTrajectoryTree,
        TestDynamics,
        TestCosts,
        TestMINDPlanner
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
        
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # 返回测试结果
    return result.wasSuccessful()


if __name__ == '__main__':
    print("运行MIND重构版本基础测试")
    print("=" * 50)
    
    success = run_basic_tests()
    
    if success:
        print("\n✓ 所有基础测试通过!")
    else:
        print("\n✗ 部分测试失败!")
        
    print("=" * 50)
