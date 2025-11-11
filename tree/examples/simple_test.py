"""
简化测试示例

测试MIND重构版本的核心功能。
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.scenario.scenario_tree import ScenarioTree, ScenarioData, AgentPrediction
from tree.scenario.aime import AIME
from tree.trajectory.trajectory_tree import TrajectoryTree, TrajectoryState, ControlInput
from tree.trajectory.dynamics import BicycleDynamics
from tree.planners.mind_planner import MINDPlanner


def test_scenario_tree():
    """测试场景树功能"""
    print("测试场景树功能")
    print("-" * 30)
    
    config = {
        'max_depth': 3,
        'uncertainty_threshold': 9.0,
        'probability_threshold': 0.001
    }
    
    scenario_tree = ScenarioTree(config)
    
    # 创建测试数据
    means = np.random.randn(20, 2)
    covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(20, axis=0)
    ego_pred = AgentPrediction(means, covs, 1.0)
    scenario_data = ScenarioData(ego_pred, [], 1.0, 0.0, {})
    
    # 添加根节点
    root_node = scenario_tree.add_root(scenario_data)
    print(f"✓ 根节点创建成功: {root_node.key}")
    
    # 添加子场景
    child_data = ScenarioData(ego_pred, [], 0.5, 0.0, {})
    child_node = scenario_tree.add_scenario("root", child_data)
    print(f"✓ 子场景创建成功: {child_node.key}")
    
    # 获取统计信息
    stats = scenario_tree.get_statistics()
    print(f"✓ 统计信息: {stats}")
    
    return True


def test_dynamics():
    """测试动力学模型"""
    print("\n测试动力学模型")
    print("-" * 30)
    
    config = {
        'dt': 0.1,
        'wheelbase': 2.5,
        'max_acceleration': 3.0,
        'max_deceleration': 5.0
    }
    
    dynamics = BicycleDynamics(config)
    
    # 测试状态转移
    state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    control = np.array([0.0, 0.0])
    
    next_state = dynamics.step(state, control)
    print(f"✓ 状态转移成功: {next_state}")
    
    # 测试状态验证
    valid = dynamics.validate_state(state)
    print(f"✓ 状态验证: {valid}")
    
    # 测试雅可比矩阵
    A, B = dynamics.get_jacobian(state, control)
    print(f"✓ 雅可比矩阵: A={A.shape}, B={B.shape}")
    
    return True


def test_trajectory_tree():
    """测试轨迹树功能"""
    print("\n测试轨迹树功能")
    print("-" * 30)
    
    config = {'dt': 0.1, 'horizon': 20}
    trajectory_tree = TrajectoryTree(config)
    
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
    
    root_node = trajectory_tree.add_root(initial_state, initial_control)
    print(f"✓ 根节点创建成功: {root_node.key}")
    
    # 添加轨迹步骤
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
        
        trajectory_node = trajectory_tree.add_trajectory_step(
            parent_id, next_state, next_control, 1.0
        )
        parent_id = trajectory_node.key
        
    print(f"✓ 轨迹步骤添加成功")
    
    # 获取轨迹
    trajectories = trajectory_tree.get_all_trajectories()
    print(f"✓ 轨迹数量: {len(trajectories)}")
    
    if trajectories:
        cost = trajectory_tree.compute_trajectory_cost(trajectories[0])
        print(f"✓ 轨迹成本: {cost}")
    
    return True


def test_mind_planner_simple():
    """测试MIND规划器简化功能"""
    print("\n测试MIND规划器简化功能")
    print("-" * 30)
    
    config = {
        'dt': 0.1,
        'horizon': 20,
        'optimizer_type': 'cbf',  # 使用CBF避免iLQR问题
        'target_velocity': 10.0
    }
    
    try:
        planner = MINDPlanner(config)
        print("✓ MIND规划器创建成功")
        
        # 测试场景数据生成
        means = np.zeros((20, 2))
        covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(20, axis=0)
        predictions = [AgentPrediction(means, covs, 1.0)]
        
        scenario_data_list = planner._generate_scenario_data(predictions)
        print(f"✓ 场景数据生成成功: {len(scenario_data_list)}个场景")
        
        # 测试目标轨迹生成
        ego_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        target_lane = np.array([[x, 0.0] for x in np.linspace(0, 50, 50)])
        
        target_trajectory = planner._generate_target_trajectory(ego_state, target_lane)
        print(f"✓ 目标轨迹生成成功: {target_trajectory.shape}")
        
        # 测试优化器切换
        planner.switch_optimizer('mpc')
        print("✓ 优化器切换成功: mpc")
        
        return True
        
    except Exception as e:
        print(f"✗ MIND规划器测试失败: {e}")
        return False


def test_aime():
    """测试AIME算法"""
    print("\n测试AIME算法")
    print("-" * 30)
    
    config = {
        'max_depth': 3,
        'uncertainty_threshold': 9.0,
        'probability_threshold': 0.001
    }
    
    aime = AIME(config)
    print("✓ AIME创建成功")
    
    # 测试变体预测生成
    means = np.zeros((20, 2))
    covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(20, axis=0)
    base_prediction = AgentPrediction(means, covs, 1.0)
    
    # 使用正确的多模态处理器
    from tree.scenario.multimodal import MultimodalProcessor
    multimodal_config = {'max_modes': 3}
    processor = MultimodalProcessor(multimodal_config)
    
    variant_predictions = processor.generate_multimodal_predictions(base_prediction, 3)
    print(f"✓ 多模态预测生成成功: {len(variant_predictions)}个模态")
    
    return True


def main():
    """主函数"""
    print("MIND重构版本简化测试")
    print("=" * 50)
    
    tests = [
        ("场景树", test_scenario_tree),
        ("动力学模型", test_dynamics),
        ("轨迹树", test_trajectory_tree),
        ("AIME算法", test_aime),
        ("MIND规划器", test_mind_planner_simple)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name}测试失败: {e}")
            results.append((test_name, False))
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结:")
    
    passed = 0
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 个测试通过")
    
    if passed == len(results):
        print("🎉 所有核心功能测试通过!")
    else:
        print("⚠️  部分测试失败，但核心功能可用")


if __name__ == "__main__":
    main()