"""
Demo 2 场景示例

基于 configs/demo_2.json 的场景演示。
目标速度: 8 m/s，适合郊区道路场景。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction


def create_demo_2_config():
    """创建Demo 2配置"""
    config = {
        'dt': 0.2,
        'state_size': 6,
        'action_size': 2,
        'horizon': 60,
        'optimizer_type': 'mpc',
        'target_velocity': 8.0,  # 从配置文件中获取
        'aime': {
            'max_depth': 5,
            'tar_dist_thres': 10.0,
            'tar_time_ahead': 5.0,
            'seg_length': 15.0,
            'seg_n_node': 10,
            'far_dist_thres': 10.0
        },
        'cost': {
            'safety': {
                'weight': 12.0,
                'safety_distance': 2.5
            },
            'target': {
                'target_velocity': 8.0,
                'weight': 8.0
            },
            'comfort': {
                'acceleration_weight': 3.0,
                'steering_weight': 2.0
            }
        },
        'constraints': {
            'state_upper_bound': np.array([100000.0, 100000.0, 12.0, 10.0, 4.0, 0.2]),
            'state_lower_bound': np.array([-100000.0, -100000.0, 0.0, -10.0, -6.0, -0.2])
        },
        'weights': {
            'w_des_state': 0.0 * np.eye(6),
            'w_ctrl': 5.0 * np.eye(2),
            'w_tgt': 1.0,
            'w_exo': 10.0,
            'w_ego': 1.0,
            'smooth_grid_res': 0.4,
            'smooth_grid_size': (256, 256)
        }
    }
    
    # 设置速度权重
    config['weights']['w_des_state'][2, 2] = 0.1
    config['weights']['w_des_state'][4, 4] = 1.0
    config['weights']['w_des_state'][5, 5] = 10.0
    
    return config


def create_suburban_road_scenario():
    """创建郊区道路场景"""
    print("创建Demo 2 - 郊区道路场景")
    print("目标速度: 8 m/s，序列ID: f4eaa49a-74a1-4829-81b2-052a650878c3")
    
    config = create_demo_2_config()
    planner = MINDPlanner(config)
    
    # 自车状态 - 郊区道路巡航
    ego_state = np.array([0.0, 0.0, 6.0, 0.0, 0.0, 0.0])
    
    predictions = []
    horizon = 60
    
    # 模态1: 正常巡航
    means1 = np.zeros((horizon, 2))
    covs1 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        # 郊区道路保持稳定速度
        speed = 6.0 + 2.0 * (1 - np.exp(-t * 0.05))  # 渐进加速到8m/s
        means1[t] = [speed * t * 0.2, 0.0]
        covs1[t] = 0.08 * np.eye(2)
    predictions.append(AgentPrediction(means1, covs1, 0.4))
    
    # 模态2: 超车场景
    means2 = np.zeros((horizon, 2))
    covs2 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 20:  # 跟车
            speed = 6.0
            means2[t] = [speed * t * 0.2, 0.0]
        elif t < 30:  # 准备超车
            speed = 8.0
            lateral_offset = 3.5 * (t - 20) / 10
            means2[t] = [speed * t * 0.2, lateral_offset]
        elif t < 40:  # 超车中
            speed = 9.0
            means2[t] = [speed * t * 0.2, 3.5]
        else:  # 返回原车道
            speed = 8.0
            lateral_offset = 3.5 * (1 - (t - 40) / 20)
            means2[t] = [speed * t * 0.2, lateral_offset]
        covs2[t] = 0.12 * np.eye(2)
    predictions.append(AgentPrediction(means2, covs2, 0.35))
    
    # 模态3: 遇到慢车减速
    means3 = np.zeros((horizon, 2))
    covs3 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 25:  # 正常行驶
            speed = 6.0 + 2.0 * (1 - np.exp(-t * 0.05))
            means3[t] = [speed * t * 0.2, 0.0]
        else:  # 遇到慢车减速
            speed = max(4.0, 8.0 - 0.2 * (t - 25))
            means3[t] = [speed * t * 0.2, 0.0]
        covs3[t] = 0.1 * np.eye(2)
    predictions.append(AgentPrediction(means3, covs3, 0.25))
    
    # 目标车道 - 直行车道
    target_lane = np.array([[x, 0.0] for x in np.linspace(0, 80, 100)])
    
    # 道路数据 - 郊区道路
    road_data = {
        'road_type': 'suburban_road',
        'speed_limit': 8.0,
        'lane_width': 3.5,
        'num_lanes': 2,
        'traffic_density': 'light',
        'features': {
            'curves': [
                {'start': 20.0, 'end': 40.0, 'radius': 50.0, 'direction': 'left'}
            ],
            'intersections': [
                {'position': np.array([60.0, 0.0]), 'type': 'stop_sign'}
            ]
        }
    }
    
    return planner, ego_state, predictions, target_lane, road_data


def visualize_demo_2_scenario():
    """可视化Demo 2场景"""
    planner, ego_state, predictions, target_lane, road_data = create_suburban_road_scenario()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demo 2 - 郊区道路场景 (目标速度: 8 m/s)', fontsize=16, fontweight='bold')
    
    # 场景概览
    ax = axes[0, 0]
    ax.set_title('场景概览', fontsize=12, fontweight='bold')
    
    colors = ['blue', 'green', 'red']
    for i, pred in enumerate(predictions):
        means = pred.means
        ax.plot(means[:, 0], means[:, 1], color=colors[i], 
                linewidth=2, label=f'模态 {i+1} (p={pred.probability:.2f})', alpha=0.7)
        
        # 绘制不确定性椭圆
        step = 12
        for t in range(0, len(means), step):
            mean = means[t]
            cov = pred.covariances[t]
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            
            from matplotlib.patches import Ellipse
            ellipse = Ellipse(mean, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]),
                            angle=np.degrees(angle), 
                            facecolor=colors[i], alpha=0.15, 
                            edgecolor=colors[i])
            ax.add_patch(ellipse)
    
    # 绘制自车
    ax.scatter(ego_state[0], ego_state[1], color='red', s=200, marker='*', 
               edgecolors='black', linewidth=2, zorder=10, label='自车')
    
    # 绘制目标车道
    ax.plot(target_lane[:, 0], target_lane[:, 1], 'k--', linewidth=2, 
            alpha=0.5, label='目标车道')
    
    # 绘制道路特征
    for curve in road_data['features']['curves']:
        ax.text(curve['start'], 5, f"转弯", fontsize=10)
    
    for intersection in road_data['features']['intersections']:
        ax.scatter(intersection['position'][0], intersection['position'][1], 
                  color='red', s=100, marker='s', label='停车标志')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 运行规划
    print("运行MIND规划器...")
    try:
        result = planner.plan(ego_state, predictions, target_lane, road_data)
        success = True
    except Exception as e:
        print(f"规划失败: {e}")
        success = False
    
    # 轨迹结果
    ax = axes[0, 1]
    ax.set_title('优化轨迹', fontsize=12, fontweight='bold')
    
    if success:
        trajectories = result['trajectory_tree'].get_all_trajectories()
        for i, traj in enumerate(trajectories):
            positions = []
            for node in traj:
                positions.append(node.trajectory_data.state.position)
            
            positions = np.array(positions)
            if len(positions) > 0:
                ax.plot(positions[:, 0], positions[:, 1], 
                       color=colors[i % len(colors)], alpha=0.6, linewidth=2,
                       label=f'轨迹 {i+1}')
        
        # 高亮最优轨迹
        best_traj = result['best_trajectory']
        if best_traj:
            positions = []
            for node in best_traj:
                positions.append(node.trajectory_data.state.position)
            
            positions = np.array(positions)
            if len(positions) > 0:
                ax.plot(positions[:, 0], positions[:, 1], 
                       color='red', linewidth=3, label='最优轨迹', zorder=10)
    else:
        ax.text(0.5, 0.5, '规划失败', transform=ax.transAxes,
               ha='center', va='center', fontsize=14, color='red')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 速度曲线
    ax = axes[1, 0]
    ax.set_title('速度曲线', fontsize=12, fontweight='bold')
    
    if success and best_traj:
        velocities = []
        times = []
        for node in best_traj:
            velocities.append(node.trajectory_data.state.velocity)
            times.append(node.trajectory_data.state.timestamp)
        
        ax.plot(times, velocities, 'b-', linewidth=2)
        ax.axhline(y=8.0, color='r', linestyle='--', label='目标速度')
        ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 统计信息
    ax = axes[1, 1]
    ax.set_title('统计信息', fontsize=12, fontweight='bold')
    
    if success:
        stats = result['statistics']
        text = f"""Demo 2 统计信息:
        
序列ID: f4eaa49a-74a1-4829-81b2-052a650878c3
目标速度: 8.0 m/s
场景类型: 郊区道路

场景树节点数: {stats['scenario_stats']['total_nodes']}
场景树最大深度: {stats['scenario_stats']['max_depth']}
轨迹树分支数: {len(result['trajectory_tree'].get_all_trajectories())}
最优轨迹成本: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.2f}

道路特征:
- 道路类型: {road_data['road_type']}
- 限速: {road_data['speed_limit']} m/s
- 车道数: {road_data['num_lanes']}
- 交通密度: {road_data['traffic_density']}
- 弯道数: {len(road_data['features']['curves'])}
"""
    else:
        text = "规划失败，无统计信息"
    
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    filename = os.path.join(plot_dir, 'demo_2_scenario.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Demo 2 场景可视化已保存到 {filename}")
    
    return result if success else None


def main():
    """主函数"""
    print("MIND Algorithm - Demo 2 场景演示")
    print("=" * 60)
    print("配置: configs/demo_2.json")
    print("序列ID: f4eaa49a-74a1-4829-81b2-052a650878c3")
    print("目标速度: 8 m/s")
    print("=" * 60)
    
    # 创建并运行场景
    result = visualize_demo_2_scenario()
    
    if result:
        print("\nDemo 2 场景运行成功!")
        
        # 测试不同优化器
        print("\n测试不同优化器性能:")
        optimizers = ['ilqr', 'mpc', 'cbf']
        
        for opt in optimizers:
            try:
                planner, ego_state, predictions, target_lane, road_data = create_suburban_road_scenario()
                planner.switch_optimizer(opt)
                
                import time
                start = time.time()
                res = planner.plan(ego_state, predictions, target_lane, road_data)
                end = time.time()
                
                control = planner.get_next_control(res)
                print(f"  {opt.upper()}: 成功, 耗时 {end-start:.3f}s, 控制 {control}")
            except Exception as e:
                print(f"  {opt.upper()}: 失败 - {e}")
    else:
        print("\nDemo 2 场景运行失败!")
    
    print("\nDemo 2 演示完成!")


if __name__ == "__main__":
    main()