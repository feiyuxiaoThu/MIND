"""
Demo 4 场景示例

基于 configs/demo_4.json 的场景演示。
目标速度: 8 m/s，适合复杂城市交叉口场景。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction


def create_demo_4_config():
    """创建Demo 4配置"""
    config = {
        'dt': 0.2,
        'state_size': 6,
        'action_size': 2,
        'horizon': 60,
        'optimizer_type': 'mpc',
        'target_velocity': 8.0,  # 从配置文件中获取
        'aime': {
            'max_depth': 6,  # 更深的搜索用于复杂场景
            'tar_dist_thres': 10.0,
            'tar_time_ahead': 5.0,
            'seg_length': 15.0,
            'seg_n_node': 10,
            'far_dist_thres': 10.0
        },
        'cost': {
            'safety': {
                'weight': 20.0,  # 更高的安全权重
                'safety_distance': 3.0
            },
            'target': {
                'target_velocity': 8.0,
                'weight': 8.0
            },
            'comfort': {
                'acceleration_weight': 2.5,
                'steering_weight': 1.5
            }
        },
        'constraints': {
            'state_upper_bound': np.array([100000.0, 100000.0, 10.0, 10.0, 3.0, 0.2]),
            'state_lower_bound': np.array([-100000.0, -100000.0, 0.0, -10.0, -5.0, -0.2])
        },
        'weights': {
            'w_des_state': 0.0 * np.eye(6),
            'w_ctrl': 5.0 * np.eye(2),
            'w_tgt': 1.0,
            'w_exo': 15.0,  # 更高的外部智能体权重
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


def create_complex_intersection_scenario():
    """创建复杂交叉口场景"""
    print("创建Demo 4 - 复杂交叉口场景")
    print("目标速度: 8 m/s，序列ID: 624a047f-598b-4d2f-ba4b-27e6699896dc")
    
    config = create_demo_4_config()
    planner = MINDPlanner(config)
    
    # 自车状态 - 接近复杂交叉口
    ego_state = np.array([-20.0, -15.0, 6.0, np.pi/4, 0.0, 0.0])
    
    predictions = []
    horizon = 60
    
    # 模态1: 直行通过交叉口
    means1 = np.zeros((horizon, 2))
    covs1 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 25:  # 接近阶段
            speed = 6.0
            means1[t] = [-20.0 + speed * t * 0.2, -15.0 + speed * t * 0.2]
        else:  # 通过交叉口
            speed = 8.0
            means1[t] = [-20.0 + 6.0 * 25 * 0.2 + speed * (t-25) * 0.2, 
                        -15.0 + 6.0 * 25 * 0.2 + speed * (t-25) * 0.2]
        covs1[t] = 0.15 * np.eye(2)
    predictions.append(AgentPrediction(means1, covs1, 0.3))
    
    # 模态2: 左转通过交叉口
    means2 = np.zeros((horizon, 2))
    covs2 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 25:  # 接近阶段
            speed = 6.0
            means2[t] = [-20.0 + speed * t * 0.2, -15.0 + speed * t * 0.2]
        else:  # 左转
            turn_progress = (t - 25) / 20
            turn_angle = np.pi/2 * turn_progress
            radius = 10.0
            center_x = -20.0 + 6.0 * 25 * 0.2 + radius
            center_y = -15.0 + 6.0 * 25 * 0.2
            means2[t] = [center_x - radius * np.sin(turn_angle),
                        center_y + radius * (1 - np.cos(turn_angle))]
        covs2[t] = 0.2 * np.eye(2)
    predictions.append(AgentPrediction(means2, covs2, 0.25))
    
    # 模态3: 遇到行人紧急制动
    means3 = np.zeros((horizon, 2))
    covs3 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 20:  # 正常接近
            speed = 6.0
            means3[t] = [-20.0 + speed * t * 0.2, -15.0 + speed * t * 0.2]
        elif t < 25:  # 制动
            speed = max(0.0, 6.0 - 3.0 * (t - 20))
            means3[t] = [means3[t-1, 0] + speed * 0.2, means3[t-1, 1] + speed * 0.2]
        elif t < 35:  # 等待
            means3[t] = means3[24]
        else:  # 重新启动
            speed = 4.0 * (t - 35) / 25
            means3[t] = [means3[24, 0] + speed * (t-35) * 0.2, 
                        means3[24, 1] + speed * (t-35) * 0.2]
        covs3[t] = 0.12 * np.eye(2)
    predictions.append(AgentPrediction(means3, covs3, 0.25))
    
    # 模态4: 右转通过交叉口
    means4 = np.zeros((horizon, 2))
    covs4 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 25:  # 接近阶段
            speed = 6.0
            means4[t] = [-20.0 + speed * t * 0.2, -15.0 + speed * t * 0.2]
        else:  # 右转
            turn_progress = (t - 25) / 15
            turn_angle = -np.pi/2 * turn_progress
            radius = 8.0
            center_x = -20.0 + 6.0 * 25 * 0.2
            center_y = -15.0 + 6.0 * 25 * 0.2 - radius
            means4[t] = [center_x + radius * np.sin(turn_angle),
                        center_y + radius * (1 - np.cos(turn_angle))]
        covs4[t] = 0.18 * np.eye(2)
    predictions.append(AgentPrediction(means4, covs4, 0.2))
    
    # 目标轨迹 - 通过交叉口
    target_trajectory = []
    for t in range(100):
        x = -20 + t * 1.0
        y = -15 + t * 1.0
        target_trajectory.append([x, y])
    target_lane = np.array(target_trajectory)
    
    # 道路数据 - 复杂交叉口
    road_data = {
        'road_type': 'complex_intersection',
        'speed_limit': 8.0,
        'intersection_center': np.array([0.0, 0.0]),
        'intersection_type': 'signalized_complex',
        'traffic_lights': {
            'current_state': 'yellow',
            'time_to_change': 3.0,
            'phases': [
                {'direction': 'NS', 'state': 'green', 'remaining': 15.0},
                {'direction': 'EW', 'state': 'red', 'remaining': 20.0}
            ]
        },
        'crosswalks': [
            {'start': np.array([-5, -5]), 'end': np.array([5, -5]), 'active': True},
            {'start': np.array([-5, 5]), 'end': np.array([5, 5]), 'active': False},
            {'start': np.array([-5, -5]), 'end': np.array([-5, 5]), 'active': False},
            {'start': np.array([5, -5]), 'end': np.array([5, 5]), 'active': False}
        ],
        'pedestrians': [
            {'position': np.array([0.0, -3.0]), 'velocity': np.array([0.5, 0.0]), 'waiting': False},
            {'position': np.array([2.0, 5.0]), 'velocity': np.array([0.0, -0.6]), 'crossing': True},
            {'position': np.array([-3.0, 0.0]), 'velocity': np.array([0.8, 0.0]), 'crossing': True}
        ],
        'surrounding_vehicles': [
            {'position': np.array([10.0, 10.0]), 'velocity': 7.0, 'direction': 'south'},
            {'position': np.array([-10.0, 5.0]), 'velocity': 6.0, 'direction': 'east'},
            {'position': np.array([5.0, -10.0]), 'velocity': 8.0, 'direction': 'north'}
        ]
    }
    
    return planner, ego_state, predictions, target_lane, road_data


def visualize_demo_4_scenario():
    """可视化Demo 4场景"""
    planner, ego_state, predictions, target_lane, road_data = create_complex_intersection_scenario()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demo 4 - 复杂交叉口场景 (目标速度: 8 m/s)', fontsize=16, fontweight='bold')
    
    # 场景概览
    ax = axes[0, 0]
    ax.set_title('场景概览', fontsize=12, fontweight='bold')
    
    colors = ['blue', 'green', 'red', 'orange']
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
            alpha=0.5, label='目标轨迹')
    
    # 绘制交叉路口
    intersection_size = 20
    rect = plt.Rectangle((-intersection_size/2, -intersection_size/2), 
                        intersection_size, intersection_size,
                        facecolor='lightgray', alpha=0.3, edgecolor='black')
    ax.add_patch(rect)
    
    # 绘制人行横道
    for crosswalk in road_data['crosswalks']:
        if crosswalk['active']:
            ax.plot([crosswalk['start'][0], crosswalk['end'][0]], 
                   [crosswalk['start'][1], crosswalk['end'][1]], 
                   'white', linewidth=3, alpha=0.8)
            ax.plot([crosswalk['start'][0], crosswalk['end'][0]], 
                   [crosswalk['start'][1], crosswalk['end'][1]], 
                   'black', linewidth=1, linestyle='--')
    
    # 绘制行人
    for ped in road_data['pedestrians']:
        ax.scatter(ped['position'][0], ped['position'][1], 
                  color='orange', s=50, marker='o', zorder=5)
        if ped['velocity'] is not None:
            ax.arrow(ped['position'][0], ped['position'][1],
                    ped['velocity'][0]*2, ped['velocity'][1]*2,
                    head_width=0.5, head_length=0.3, fc='orange', ec='orange', alpha=0.7)
    
    # 绘制周围车辆
    for vehicle in road_data['surrounding_vehicles']:
        ax.scatter(vehicle['position'][0], vehicle['position'][1], 
                  color='gray', s=100, marker='s', zorder=5)
    
    # 绘制交通灯
    light_pos = np.array([15.0, 15.0])
    ax.scatter(light_pos[0], light_pos[1], color='yellow', s=150, marker='h', 
              edgecolors='black', linewidth=2, label='交通灯')
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    
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
    
    # 重新绘制交叉路口
    rect = plt.Rectangle((-intersection_size/2, -intersection_size/2), 
                        intersection_size, intersection_size,
                        facecolor='lightgray', alpha=0.3, edgecolor='black')
    ax.add_patch(rect)
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-30, 30)
    ax.set_ylim(-30, 30)
    
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
        text = f"""Demo 4 统计信息:
        
序列ID: 624a047f-598b-4d2f-ba4b-27e6699896dc
目标速度: 8.0 m/s
场景类型: 复杂交叉口

场景树节点数: {stats['scenario_stats']['total_nodes']}
场景树最大深度: {stats['scenario_stats']['max_depth']}
轨迹树分支数: {len(result['trajectory_tree'].get_all_trajectories())}
最优轨迹成本: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.2f}

道路特征:
- 交叉口类型: {road_data['intersection_type']}
- 限速: {road_data['speed_limit']} m/s
- 交通灯状态: {road_data['traffic_lights']['current_state']}
- 人行横道数: {len(road_data['crosswalks'])}
- 行人数: {len(road_data['pedestrians'])}
- 周围车辆数: {len(road_data['surrounding_vehicles'])}
"""
    else:
        text = "规划失败，无统计信息"
    
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=9, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.axis('off')
    
    plt.tight_layout()
    
    # 保存结果
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    filename = os.path.join(plot_dir, 'demo_4_scenario.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Demo 4 场景可视化已保存到 {filename}")
    
    return result if success else None


def main():
    """主函数"""
    print("MIND Algorithm - Demo 4 场景演示")
    print("=" * 60)
    print("配置: configs/demo_4.json")
    print("序列ID: 624a047f-598b-4d2f-ba4b-27e6699896dc")
    print("目标速度: 8 m/s")
    print("=" * 60)
    
    # 创建并运行场景
    result = visualize_demo_4_scenario()
    
    if result:
        print("\nDemo 4 场景运行成功!")
        
        # 测试不同优化器
        print("\n测试不同优化器性能:")
        optimizers = ['ilqr', 'mpc', 'cbf']
        
        for opt in optimizers:
            try:
                planner, ego_state, predictions, target_lane, road_data = create_complex_intersection_scenario()
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
        print("\nDemo 4 场景运行失败!")
    
    print("\nDemo 4 演示完成!")


if __name__ == "__main__":
    main()