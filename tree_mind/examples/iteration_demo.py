"""
MIND算法迭代演示

展示算法在不同迭代阶段的表现。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction
from tree.scenarios.scenario_factory import ScenarioFactory


def create_demonstration_scenarios():
    """创建演示场景"""
    scenarios = []
    
    # 场景1: 简单直行
    print("创建场景1: 简单直行")
    config1 = {
        'dt': 0.1,
        'horizon': 20,
        'optimizer_type': 'cbf',
        'target_velocity': 10.0,
        'cost': {
            'target': {'target_velocity': 10.0, 'position_weight': 1.0},
            'safety': {'weight': 5.0}
        }
    }
    
    planner1 = MINDPlanner(config1)
    ego_state1 = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    
    # 单一直行预测
    predictions1 = []
    horizon = 20
    means = np.zeros((horizon, 2))
    covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(horizon, axis=0)
    
    for t in range(horizon):
        means[t] = [10.0 * t * 0.1, 0.0]
        
    predictions1.append(AgentPrediction(means, covs, 1.0))
    
    target_lane1 = np.array([[x, 0.0] for x in np.linspace(0, 30, 100)])
    road_data1 = {'road_type': 'highway'}
    
    scenarios.append({
        'name': '简单直行',
        'planner': planner1,
        'ego_state': ego_state1,
        'predictions': predictions1,
        'target_lane': target_lane1,
        'road_data': road_data1
    })
    
    # 场景2: 换道场景
    print("创建场景2: 换道场景")
    config2 = {
        'dt': 0.1,
        'horizon': 30,
        'optimizer_type': 'mpc',
        'target_velocity': 12.0,
        'cost': {
            'target': {'target_velocity': 12.0},
            'comfort': {'weight': 2.0}
        }
    }
    
    planner2 = MINDPlanner(config2)
    ego_state2 = np.array([0.0, 0.0, 12.0, 0.0, 0.0, 0.0])
    
    # 换道预测
    predictions2 = []
    for strategy in ['aggressive', 'normal', 'conservative']:
        means = np.zeros((30, 2))
        covs = 0.15 * np.eye(2)[np.newaxis, :, :].repeat(30, axis=0)
        
        for t in range(30):
            if strategy == 'aggressive':
                lateral_offset = 3.5 * (t / 30)  # 快速换道
            elif strategy == 'conservative':
                lateral_offset = 3.5 * (t / 30) * 0.7  # 慢速换道
            else:
                lateral_offset = 3.5 * (t / 30)  # 正常换道
                
            means[t] = [12.0 * t * 0.1, lateral_offset]
            
        predictions2.append(AgentPrediction(means, covs, 1.0/3))
    
    target_lane2 = np.array([[x, 3.5] for x in np.linspace(0, 50, 100)])
    road_data2 = {
        'current_lane': 'lane_1',
        'target_lane': 'lane_2',
        'road_type': 'highway'
    }
    
    scenarios.append({
        'name': '换道场景',
        'planner': planner2,
        'ego_state': ego_state2,
        'predictions': predictions2,
        'target_lane': target_lane2,
        'road_data': road_data2
    })
    
    # 场景3: 路口场景
    print("创建场景3: 路口场景")
    config3 = {
        'dt': 0.1,
        'horizon': 40,
        'optimizer_type': 'mpc',
        'target_velocity': 8.0,
        'cost': {
            'safety': {'weight': 15.0, 'safety_distance': 3.0},
            'target': {'target_velocity': 8.0}
        }
    }
    
    planner3 = MINDPlanner(config3)
    ego_state3 = np.array([-15.0, -10.0, 8.0, np.pi/4, 0.0, 0.0])
    
    # 路口多模态预测
    predictions3 = []
    horizon = 40
    
    # 模态1: 直行
    means1 = np.zeros((horizon, 2))
    covs1 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        means1[t] = [-15.0 + 8.0 * t * 0.1, -10.0 + 8.0 * t * 0.1]
        covs1[t] = 0.2 * np.eye(2)
    predictions3.append(AgentPrediction(means1, covs1, 0.4))
    
    # 模态2: 左转
    means2 = np.zeros((horizon, 2))
    covs2 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        means2[t] = [-15.0 + 6.0 * t * 0.1, -10.0 + 6.0 * t * 0.1 - 0.3 * t * 0.1**2]
        covs2[t] = 0.3 * np.eye(2)
    predictions3.append(AgentPrediction(means2, covs2, 0.3))
    
    # 模态3: 右转
    means3 = np.zeros((horizon, 2))
    covs3 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        means3[t] = [-15.0 + 6.0 * t * 0.1, -10.0 + 6.0 * t * 0.1 + 0.2 * t * 0.1**2]
        covs3[t] = 0.3 * np.eye(2)
    predictions3.append(AgentPrediction(means3, covs3, 0.3))
    
    target_lane3 = np.array([[-15 + x, -10 + x] for x in np.linspace(0, 40, 100)])
    road_data3 = {
        'intersection_center': np.array([0.0, 0.0]),
        'intersection_type': 'four_way',
        'road_type': 'intersection'
    }
    
    scenarios.append({
        'name': '路口场景',
        'planner': planner3,
        'ego_state': ego_state3,
        'predictions': predictions3,
        'target_lane': target_lane3,
        'road_data': road_data3
    })
    
    return scenarios


def visualize_iteration_results(scenarios):
    """可视化迭代结果"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('MIND算法迭代演示结果', fontsize=16, fontweight='bold')
    
    for i, scenario in enumerate(scenarios):
        ax = axes[i]
        ax.set_title(scenario['name'], fontsize=14, fontweight='bold')
        
        try:
            # 运行规划
            result = scenario['planner'].plan(
                scenario['ego_state'], 
                scenario['predictions'], 
                scenario['target_lane'], 
                scenario['road_data']
            )
            
            # 可视化轨迹
            trajectories = result['trajectory_tree'].get_all_trajectories()
            colors = ['blue', 'green', 'red']
            
            for j, traj in enumerate(trajectories):
                positions = []
                for node in traj:
                    positions.append(node.trajectory_data.state.position)
                
                positions = np.array(positions)
                if len(positions) > 0:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           color=colors[j % len(colors)], alpha=0.6, linewidth=2)
                    ax.scatter(positions[0, 0], positions[0, 1], 
                               color=colors[j % len(colors)], s=50, marker='o')
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                               color=colors[j % len(colors)], s=50, marker='s')
            
            # 高亮最优轨迹
            best_traj = result['best_trajectory']
            if best_traj:
                positions = []
                for node in best_traj:
                    positions.append(node.trajectory_data.state.position)
                
                positions = np.array(positions)
                if len(positions) > 0:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           color='red', linewidth=3, zorder=10)
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                               color='red', s=100, marker='*', zorder=10)
            
            # 绘制目标车道
            target_lane = scenario['target_lane']
            ax.plot(target_lane[:, 0], target_lane[:, 1], 'k--', alpha=0.3, label='目标车道')
            
            # 添加统计信息
            stats = result['statistics']
            info_text = f"""场景数: {stats['scenario_stats']['total_nodes']}
轨迹数: {len(trajectories)}
最优成本: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.1f}"""
            
            ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        except Exception as e:
            ax.text(0.5, 0.5, f'规划失败:\n{str(e)}', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, color='red')
        
        ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return fig


def analyze_performance(scenarios):
    """分析性能"""
    print("\n性能分析:")
    print("=" * 50)
    
    for scenario in scenarios:
        print(f"\n{scenario['name']}:")
        
        try:
            # 运行规划
            import time
            start_time = time.time()
            result = scenario['planner'].plan(
                scenario['ego_state'], 
                scenario['predictions'], 
                scenario['target_lane'], 
                scenario['road_data']
            )
            end_time = time.time()
            
            # 收集统计信息
            stats = result['statistics']
            best_traj = result['best_trajectory']
            
            print(f"  规划时间: {end_time - start_time:.3f}秒")
            print(f"  场景树节点数: {stats['scenario_stats']['total_nodes']}")
            print(f"  场景树最大深度: {stats['scenario_stats']['max_depth']}")
            print(f"  轨迹树分支数: {len(result['trajectory_tree'].get_all_trajectories())}")
            print(f"  最优轨迹成本: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.2f}")
            
            # 验证规划
            validation = scenario['planner'].validate_plan(result)
            print(f"  规划验证: {'通过' if validation['valid'] else '失败'}")
            
            # 获取控制
            next_control = scenario['planner'].get_next_control(result)
            print(f"  下一控制: 加速度={next_control[0]:.2f}, 转向率={next_control[1]:.3f}")
            
        except Exception as e:
            print(f"  规划失败: {e}")
    
    print("\n" + "=" * 50)


def compare_optimizers():
    """比较不同优化器"""
    print("\n优化器比较:")
    print("=" * 50)
    
    # 创建测试场景
    config = {
        'dt': 0.1,
        'horizon': 25,
        'target_velocity': 10.0
    }
    
    ego_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
    
    # 创建预测
    predictions = []
    horizon = 25
    means = np.zeros((horizon, 2))
    covs = 0.1 * np.eye(2)[np.newaxis, :, :].repeat(horizon, axis=0)
    
    for t in range(horizon):
        means[t] = [10.0 * t * 0.1, 0.0]
        
    predictions.append(AgentPrediction(means, covs, 1.0))
    
    target_lane = np.array([[x, 0.0] for x in np.linspace(0, 30, 100)])
    road_data = {'road_type': 'highway'}
    
    optimizers = ['cbf', 'mpc']
    
    for optimizer_type in optimizers:
        print(f"\n{optimizer_type.upper()} 优化器:")
        
        try:
            config['optimizer_type'] = optimizer_type
            planner = MINDPlanner(config)
            
            import time
            start_time = time.time()
            result = planner.plan(ego_state, predictions, target_lane, road_data)
            end_time = time.time()
            
            stats = result['statistics']
            best_traj = result['best_trajectory']
            
            print(f"  执行时间: {end_time - start_time:.3f}秒")
            print(f"  轨迹数: {len(result['trajectory_tree'].get_all_trajectories())}")
            print(f"  最优成本: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.2f}")
            
            validation = planner.validate_plan(result)
            print(f"  验证结果: {'通过' if validation['valid'] else '失败'}")
            
        except Exception as e:
            print(f"  执行失败: {e}")


def main():
    """主函数"""
    print("MIND算法迭代演示")
    print("=" * 50)
    
    # 创建演示场景
    scenarios = create_demonstration_scenarios()
    
    # 可视化结果
    fig = visualize_iteration_results(scenarios)
    
    # 保存结果
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    filename = os.path.join(plot_dir, 'iteration_results.png')
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"\n可视化结果已保存到 {filename}")
    
    # 性能分析
    analyze_performance(scenarios)
    
    # 优化器比较
    compare_optimizers()
    
    print("\n演示完成！")
    print("生成的文件:")
    print("- iteration_results.png: 场景迭代可视化结果")


if __name__ == "__main__":
    main()
