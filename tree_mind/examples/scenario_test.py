"""
场景测试示例

测试不同场景下的MIND算法性能。
"""

import numpy as np
import sys
import os
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction
from tree.scenarios.scenario_factory import ScenarioFactory


def create_intersection_test():
    """创建路口测试场景"""
    print("测试路口场景")
    print("-" * 30)
    
    config = {
        'aime': {'max_depth': 6, 'uncertainty_threshold': 8.0},
        'cost': {
            'safety': {'weight': 10.0, 'safety_distance': 3.0},
            'target': {'target_velocity': 8.0}
        },
        'dt': 0.1, 'horizon': 60
    }
    
    planner = MINDPlanner(config)
    
    # 路口场景状态
    ego_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    
    # 路口预测（冲突车辆）
    predictions = []
    horizon = 60
    
    # 自车预测（直行通过路口）
    ego_means = np.zeros((horizon, 2))
    ego_covs = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        ego_means[t] = [10.0 * t * 0.1, 0.0]
        ego_covs[t] = 0.2 * np.eye(2)
    predictions.append(AgentPrediction(ego_means, ego_covs, 0.5))
    
    # 冲突车辆预测（从左侧进入）
    conflict_means = np.zeros((horizon, 2))
    conflict_covs = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        conflict_means[t] = [-5.0 + 8.0 * t * 0.1, 10.0]
        conflict_covs[t] = 0.3 * np.eye(2)
    predictions.append(AgentPrediction(conflict_means, conflict_covs, 0.5))
    
    # 路口道路数据
    road_data = {
        'intersection_center': np.array([0.0, 0.0]),
        'intersection_type': 'four_way',
        'road_type': 'intersection'
    }
    
    # 目标车道（直行）
    target_lane = np.array([[x, 0.0] for x in np.linspace(0, 100, 100)])
    
    # 执行规划
    result = planner.plan(ego_state, predictions, target_lane, road_data)
    
    # 输出结果
    print(f"路口场景规划完成")
    print(f"最优轨迹长度: {len(result['best_trajectory'])}")
    print(f"场景树节点数: {result['statistics']['scenario_stats']['total_nodes']}")
    
    # 验证结果
    validation = planner.validate_plan(result)
    print(f"规划验证: {'通过' if validation['valid'] else '失败'}")
    
    return result


def create_lane_change_test():
    """创建换道测试场景"""
    print("\n测试换道场景")
    print("-" * 30)
    
    config = {
        'cost': {
            'comfort': {'weight': 2.0},
            'target': {'target_velocity': 15.0}
        },
        'dt': 0.1, 'horizon': 80
    }
    
    planner = MINDPlanner(config)
    
    # 换道场景状态
    ego_state = np.array([0.0, 0.0, 15.0, 0.0, 0.0, 0.0])
    
    # 换道预测
    predictions = []
    horizon = 80
    
    # 自车预测（准备换道）
    for strategy in ['aggressive', 'normal', 'conservative']:
        means = np.zeros((horizon, 2))
        covs = np.zeros((horizon, 2, 2))
        
        for t in range(horizon):
            # 横向偏移（换道轨迹）
            if t < 40:  # 前4秒准备
                lateral_offset = 0.1 * t / 40
            else:  # 后4秒换道
                lateral_offset = 0.1 + 3.4 * (t - 40) / 40
                
            means[t] = [15.0 * t * 0.1, lateral_offset]
            covs[t] = 0.15 * np.eye(2)
            
        predictions.append(AgentPrediction(means, covs, 1.0/3.0))
    
    # 换道道路数据
    road_data = {
        'current_lane': 'lane_1',
        'target_lane': 'lane_2',
        'current_lane_center': np.array([[x, 0.0] for x in np.linspace(0, 100, 100)]),
        'target_lane_center': np.array([[x, 3.5] for x in np.linspace(0, 100, 100)])
    }
    
    # 目标车道（目标车道中心）
    target_lane = road_data['target_lane_center']
    
    # 执行规划
    result = planner.plan(ego_state, predictions, target_lane, road_data)
    
    # 输出结果
    print(f"换道场景规划完成")
    print(f"最优轨迹长度: {len(result['best_trajectory'])}")
    print(f"场景树最大深度: {result['statistics']['scenario_stats']['max_depth']}")
    
    # 验证结果
    validation = planner.validate_plan(result)
    print(f"规划验证: {'通过' if validation['valid'] else '失败'}")
    
    return result


def create_highway_test():
    """创建高速测试场景"""
    print("\n测试高速场景")
    print("-" * 30)
    
    config = {
        'cost': {
            'safety': {'weight': 5.0},
            'target': {'target_velocity': 25.0}
        },
        'dt': 0.1, 'horizon': 50
    }
    
    planner = MINDPlanner(config)
    
    # 高速场景状态
    ego_state = np.array([0.0, 0.0, 25.0, 0.0, 0.0, 0.0])
    
    # 高速预测（多车辆场景）
    predictions = []
    horizon = 50
    
    # 自车预测（保持车道）
    ego_means = np.zeros((horizon, 2))
    ego_covs = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        ego_means[t] = [25.0 * t * 0.1, 0.0]
        ego_covs[t] = 0.1 * np.eye(2)
    predictions.append(AgentPrediction(ego_means, ego_covs, 0.4))
    
    # 前方车辆（较慢）
    front_means = np.zeros((horizon, 2))
    front_covs = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        front_means[t] = [50.0 + 20.0 * t * 0.1, 0.0]
        front_covs[t] = 0.15 * np.eye(2)
    predictions.append(AgentPrediction(front_means, front_covs, 0.3))
    
    # 旁边车辆（潜在威胁）
    side_means = np.zeros((horizon, 2))
    side_covs = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        side_means[t] = [25.0 * t * 0.1, 3.5]
        side_covs[t] = 0.2 * np.eye(2)
    predictions.append(AgentPrediction(side_means, side_covs, 0.3))
    
    # 高速道路数据
    road_data = {
        'road_type': 'highway',
        'num_lanes': 3,
        'traffic_density': 'moderate'
    }
    
    # 目标车道（当前车道）
    target_lane = np.array([[x, 0.0] for x in np.linspace(0, 100, 100)])
    
    # 执行规划
    result = planner.plan(ego_state, predictions, target_lane, road_data)
    
    # 输出结果
    print(f"高速场景规划完成")
    print(f"最优轨迹长度: {len(result['best_trajectory'])}")
    print(f"轨迹树分支数: {len(result['trajectory_tree'].get_all_trajectories())}")
    
    # 验证结果
    validation = planner.validate_plan(result)
    print(f"规划验证: {'通过' if validation['valid'] else '失败'}")
    
    return result


def visualize_results(results, titles):
    """可视化结果"""
    try:
        fig, axes = plt.subplots(1, len(results), figsize=(15, 5))
        
        if len(results) == 1:
            axes = [axes]
            
        for i, (result, title) in enumerate(zip(results, titles)):
            ax = axes[i]
            
            # 提取轨迹
            best_trajectory = result['best_trajectory']
            if best_trajectory:
                positions = np.array([node.trajectory_data.state.position 
                                    for node in best_trajectory])
                
                ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='最优轨迹')
                ax.scatter(positions[0, 0], positions[0, 1], c='g', s=100, label='起点')
                ax.scatter(positions[-1, 0], positions[-1, 1], c='r', s=100, label='终点')
                
            ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True)
    ax.axis('equal')
            
        plt.tight_layout()
        plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    plt.savefig(os.path.join(plot_dir, 'scenario_test_results.png'))
        print(f"\n可视化结果已保存到 plot/scenario_test_results.png")
        
    except Exception as e:
        print(f"可视化失败: {e}")


def main():
    """主函数"""
    print("MIND重构版本场景测试")
    print("=" * 50)
    
    results = []
    titles = []
    
    try:
        # 测试路口场景
        intersection_result = create_intersection_test()
        results.append(intersection_result)
        titles.append('路口场景')
        
        # 测试换道场景
        lane_change_result = create_lane_change_test()
        results.append(lane_change_result)
        titles.append('换道场景')
        
        # 测试高速场景
        highway_result = create_highway_test()
        results.append(highway_result)
        titles.append('高速场景')
        
        # 可视化结果
        visualize_results(results, titles)
        
        print("\n所有场景测试完成!")
        
    except Exception as e:
        print(f"测试执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()