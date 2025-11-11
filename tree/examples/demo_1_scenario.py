"""
Demo 1 场景示例

基于 configs/demo_1.json 的场景演示。
目标速度: 4 m/s，适合城市道路场景。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import matplotlib.font_manager as fm

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction


def create_demo_1_config():
    """创建Demo 1配置"""
    config = {
        'dt': 0.2,
        'state_size': 6,
        'action_size': 2,
        'horizon': 50,
        'optimizer_type': 'mpc',
        'target_velocity': 4.0,  # 从配置文件中获取
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
                'weight': 15.0,
                'safety_distance': 3.0
            },
            'target': {
                'target_velocity': 4.0,
                'weight': 5.0
            },
            'comfort': {
                'acceleration_weight': 2.0,
                'steering_weight': 1.0
            }
        },
        'constraints': {
            'state_upper_bound': np.array([100000.0, 100000.0, 8.0, 10.0, 4.0, 0.2]),
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


def create_urban_driving_scenario():
    """创建城市驾驶场景"""
    print("创建Demo 1 - 城市驾驶场景")
    print("目标速度: 4 m/s，序列ID: 24520ce8-038f-4e5e-a455-8c06877504ab")
    
    config = create_demo_1_config()
    planner = MINDPlanner(config)
    
    # 自车状态 - 城市道路起步
    ego_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    
    predictions = []
    horizon = 50
    
    # 模态1: 正常起步行驶
    means1 = np.zeros((horizon, 2))
    covs1 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        # 城市道路加速到目标速度
        speed = min(4.0, 2.0 * t * 0.2)
        means1[t] = [speed * t * 0.2, 0.0]
        covs1[t] = 0.1 * np.eye(2)
    predictions.append(AgentPrediction(means1, covs1, 0.4))
    
    # 模态2: 谨慎起步（考虑行人）
    means2 = np.zeros((horizon, 2))
    covs2 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 10:  # 等待
            means2[t] = [0.0, 0.0]
            covs2[t] = 0.05 * np.eye(2)
        else:  # 缓慢起步
            speed = min(3.0, 1.5 * (t-10) * 0.2)
            means2[t] = [speed * (t-10) * 0.2, 0.0]
            covs2[t] = 0.15 * np.eye(2)
    predictions.append(AgentPrediction(means2, covs2, 0.35))
    
    # 模态3: 遇到红灯停车
    means3 = np.zeros((horizon, 2))
    covs3 = np.zeros((horizon, 2, 2))
    for t in range(horizon):
        if t < 15:  # 正常行驶
            speed = min(4.0, 2.0 * t * 0.2)
            means3[t] = [speed * t * 0.2, 0.0]
            covs3[t] = 0.1 * np.eye(2)
        else:  # 红灯停车
            means3[t] = means3[14]
            covs3[t] = 0.05 * np.eye(2)
    predictions.append(AgentPrediction(means3, covs3, 0.25))
    
    # 目标车道 - 直行车道
    target_lane = np.array([[x, 0.0] for x in np.linspace(0, 40, 100)])
    
    # 道路数据 - 城市道路
    road_data = {
        'road_type': 'urban_road',
        'speed_limit': 4.0,
        'lane_width': 3.5,
        'traffic_light': {
            'state': 'green',
            'distance_to_light': 30.0
        },
        'crosswalk': {
            'position': np.array([20.0, 0.0]),
            'has_pedestrians': False
        }
    }
    
    return planner, ego_state, predictions, target_lane, road_data


def setup_chinese_font():
    """设置中文字体"""
    try:
        # 使用系统默认字体，避免中文字体问题
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        return True
    except:
        return False

def visualize_demo_1_scenario():
    """可视化Demo 1场景"""
    setup_chinese_font()
    planner, ego_state, predictions, target_lane, road_data = create_urban_driving_scenario()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Demo 1 - 城市驾驶场景 (目标速度: 4 m/s)', fontsize=16, fontweight='bold', fontfamily='SimHei')
    
    # 场景概览
    ax = axes[0, 0]
    ax.set_title('场景概览', fontsize=12, fontweight='bold', fontfamily='SimHei')
    
    colors = ['blue', 'green', 'red']
    for i, pred in enumerate(predictions):
        means = pred.means
        ax.plot(means[:, 0], means[:, 1], color=colors[i], 
                linewidth=2, label=f'模态 {i+1} (p={pred.probability:.2f})', alpha=0.7)
        
        # 绘制不确定性椭圆
        step = 10
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
    
    # 绘制交通灯
    ax.scatter(30.0, 2.0, color='yellow', s=100, marker='o', label='交通灯')
    
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
    ax.set_title('优化轨迹', fontsize=12, fontweight='bold', fontfamily='SimHei')
    
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
    ax.set_title('速度曲线', fontsize=12, fontweight='bold', fontfamily='SimHei')
    
    if success and best_traj:
        velocities = []
        times = []
        for node in best_traj:
            velocities.append(node.trajectory_data.state.velocity)
            times.append(node.trajectory_data.state.timestamp)
        
        ax.plot(times, velocities, 'b-', linewidth=2)
        ax.axhline(y=4.0, color='r', linestyle='--', label='目标速度')
        ax.set_xlabel('时间 (s)')
    ax.set_ylabel('速度 (m/s)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 统计信息
    ax = axes[1, 1]
    ax.set_title('统计信息', fontsize=12, fontweight='bold', fontfamily='SimHei')
    
    if success:
        stats = result['statistics']
        text = f"""Demo 1 Statistics:
        
Sequence ID: 24520ce8-038f-4e5e-a455-8c06877504ab
Target Velocity: 4.0 m/s
Scenario Type: Urban Road

Scenario Tree Nodes: {stats['scenario_stats']['total_nodes']}
Scenario Tree Max Depth: {stats['scenario_stats']['max_depth']}
Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}
Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(best_traj):.2f}

Road Features:
- Road Type: {road_data['road_type']}
- Speed Limit: {road_data['speed_limit']} m/s
- Lane Width: {road_data['lane_width']} m
- Traffic Light State: {road_data.get('traffic_light', {}).get('current_state', 'N/A')}
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
    filename = os.path.join(plot_dir, 'demo_1_scenario.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Demo 1 场景可视化已保存到 {filename}")
    
    return result if success else None


def main():
    """主函数"""
    print("MIND Algorithm - Demo 1 场景演示")
    print("=" * 60)
    print("配置: configs/demo_1.json")
    print("序列ID: 24520ce8-038f-4e5e-a455-8c06877504ab")
    print("目标速度: 4 m/s")
    print("=" * 60)
    
    # 创建并运行场景
    result = visualize_demo_1_scenario()
    
    if result:
        print("\nDemo 1 场景运行成功!")
        
        # 测试不同优化器
        print("\n测试不同优化器性能:")
        optimizers = ['ilqr', 'mpc', 'cbf']
        
        for opt in optimizers:
            try:
                planner, ego_state, predictions, target_lane, road_data = create_urban_driving_scenario()
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
        print("\nDemo 1 场景运行失败!")
    
    print("\nDemo 1 演示完成!")


if __name__ == "__main__":
    main()
