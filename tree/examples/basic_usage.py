"""
MIND重构版本基本使用示例

演示如何使用重构后的MIND算法进行轨迹规划。
"""

import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction
from tree.scenarios.scenario_factory import ScenarioFactory


def create_sample_config():
    """创建示例配置"""
    config = {
        # AIME配置
        'aime': {
            'max_depth': 5,
            'uncertainty_threshold': 9.0,
            'probability_threshold': 0.001,
            'topology_threshold': np.pi / 6
        },
        
        # 多模态配置
        'multimodal': {
            'max_modes': 6,
            'probability_threshold': 0.05,
            'diversity_threshold': 2.0
        },
        
        # 不确定性配置
        'uncertainty': {
            'min_covariance': 1e-5,
            'max_covariance': 10.0,
            'uncertainty_growth_rate': 1.1
        },
        
        # 动力学配置
        'dynamics': {
            'dt': 0.1,
            'wheelbase': 2.5,
            'max_acceleration': 3.0,
            'max_deceleration': 5.0,
            'max_steering_angle': 0.5
        },
        
        # 成本配置
        'cost': {
            'use_safety_cost': True,
            'use_target_cost': True,
            'use_comfort_cost': True,
            'safety': {
                'safety_distance': 2.0,
                'collision_penalty': 1000.0
            },
            'target': {
                'target_velocity': 10.0,
                'position_weight': 1.0,
                'velocity_weight': 0.5
            },
            'comfort': {
                'max_acceleration': 3.0,
                'max_steering_rate': 0.5
            }
        },
        
        # 优化器配置
        'optimizer': {
            'type': 'ilqr',
            'max_iterations': 100,
            'tolerance': 1e-6,
            'regularization': 1e-4
        },
        
        # 规划参数
        'dt': 0.1,
        'horizon': 50,
        'target_velocity': 10.0
    }
    
    return config


def create_sample_ego_state():
    """创建示例自车状态"""
    return np.array([
        0.0,    # x
        0.0,    # y
        10.0,   # v (m/s)
        0.0,    # theta
        0.0,    # a
        0.0     # delta
    ])


def create_sample_predictions():
    """创建示例多模态预测"""
    predictions = []
    
    # 预测视野
    horizon = 50
    dt = 0.1
    
    # 创建3种不同的预测模态
    for i in range(3):
        means = np.zeros((horizon, 2))
        covariances = np.zeros((horizon, 2, 2))
        
        # 生成轨迹
        for t in range(horizon):
            if i == 0:  # 直行
                means[t] = [10.0 * t * dt, 0.0]
            elif i == 1:  # 左转
                means[t] = [10.0 * t * dt, 0.5 * np.sin(0.1 * t)]
            else:  # 右转
                means[t] = [10.0 * t * dt, -0.5 * np.sin(0.1 * t)]
                
            # 设置协方差（随时间增长）
            base_cov = 0.1 * (1 + 0.01 * t)
            covariances[t] = base_cov * np.eye(2)
            
        prediction = AgentPrediction(
            means=means,
            covariances=covariances,
            probability=1.0 / 3.0
        )
        predictions.append(prediction)
        
    return predictions


def create_sample_target_lane():
    """创建示例目标车道"""
    lane_points = []
    for i in range(100):
        x = i * 2.0  # 每2米一个点
        y = 0.0
        lane_points.append([x, y])
        
    return np.array(lane_points)


def create_sample_road_data():
    """创建示例道路数据"""
    return {
        'road_type': 'highway',
        'num_lanes': 3,
        'current_lane': 'lane_1',
        'target_lane': 'lane_2',
        'current_lane_center': create_sample_target_lane(),
        'target_lane_center': create_sample_target_lane() + np.array([0, 3.5])  # 右车道
    }


def main():
    """主函数"""
    print("MIND重构版本基本使用示例")
    print("=" * 50)
    
    # 1. 创建配置
    config = create_sample_config()
    print("✓ 配置创建完成")
    
    # 2. 创建规划器
    planner = MINDPlanner(config)
    print("✓ MIND规划器创建完成")
    
    # 3. 创建输入数据
    ego_state = create_sample_ego_state()
    predictions = create_sample_predictions()
    target_lane = create_sample_target_lane()
    road_data = create_sample_road_data()
    
    print("✓ 输入数据创建完成")
    print(f"  - 自车状态: {ego_state}")
    print(f"  - 预测模态数: {len(predictions)}")
    print(f"  - 目标车道点数: {len(target_lane)}")
    
    # 4. 场景检测
    scenario_factory = ScenarioFactory(config)
    scenario_type = scenario_factory.detect_scenario(ego_state, road_data, predictions)
    print(f"✓ 场景检测完成: {scenario_type}")
    
    # 5. 执行规划
    print("\n开始执行规划...")
    try:
        result = planner.plan(ego_state, predictions, target_lane, road_data)
        print("✓ 规划执行完成")
        
        # 6. 提取结果
        best_trajectory = result['best_trajectory']
        next_control = planner.get_next_control(result)
        
        print(f"\n规划结果:")
        print(f"  - 最优轨迹长度: {len(best_trajectory)}")
        print(f"  - 下一控制输入: {next_control}")
        print(f"  - 场景树统计: {result['statistics']['scenario_stats']}")
        print(f"  - 轨迹树统计: {result['statistics']['trajectory_stats']}")
        
        # 7. 验证规划结果
        validation = planner.validate_plan(result)
        print(f"\n验证结果:")
        print(f"  - 规划有效: {validation['valid']}")
        if validation['issues']:
            print(f"  - 问题: {validation['issues']}")
        if validation['warnings']:
            print(f"  - 警告: {validation['warnings']}")
            
    except Exception as e:
        print(f"✗ 规划执行失败: {e}")
        import traceback
        traceback.print_exc()
        
    # 8. 测试不同优化器
    print("\n测试不同优化器...")
    optimizers = ['ilqr', 'mpc', 'cbf']
    
    for optimizer_type in optimizers:
        try:
            print(f"\n测试 {optimizer_type} 优化器:")
            planner.switch_optimizer(optimizer_type)
            
            result = planner.plan(ego_state, predictions, target_lane, road_data)
            next_control = planner.get_next_control(result)
            
            print(f"  ✓ {optimizer_type} 优化器测试成功")
            print(f"    下一控制: {next_control}")
            
        except Exception as e:
            print(f"  ✗ {optimizer_type} 优化器测试失败: {e}")
            
    print("\n示例执行完成!")


if __name__ == "__main__":
    main()