"""
MARC规划器演示

展示如何使用MARC规划器进行轨迹规划。
"""

import numpy as np
import json
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree_marc.planners.mind_planner import MARCPlanner


def load_config(config_path: str) -> dict:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def generate_initial_state() -> np.ndarray:
    """生成初始状态"""
    return np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])  # [x, y, v, theta, a, delta]


def generate_target_trajectory(horizon: int = 50) -> np.ndarray:
    """生成目标轨迹"""
    trajectory = np.zeros((horizon, 6))
    
    for t in range(horizon):
        # 简单的直线轨迹
        trajectory[t] = [
            10.0 * t * 0.1,  # x
            0.0,              # y
            10.0,             # v
            0.0,              # theta
            0.0,              # a
            0.0               # delta
        ]
        
    return trajectory


def generate_exo_agents() -> list:
    """生成外部智能体"""
    agents = []
    
    # 智能体1：前方车辆
    agent1 = {
        'id': 'vehicle_1',
        'initial_state': np.array([20.0, 0.0, 8.0, 0.0, 0.0, 0.0]),
        'trajectory': np.array([
            [20.0 + 8.0 * t * 0.1, 0.0, 8.0, 0.0, 0.0, 0.0] 
            for t in range(50)
        ]),
        'type': 'vehicle'
    }
    agents.append(agent1)
    
    # 智能体2：侧方车辆
    agent2 = {
        'id': 'vehicle_2',
        'initial_state': np.array([10.0, 5.0, 9.0, 0.0, 0.0, 0.0]),
        'trajectory': np.array([
            [10.0 + 9.0 * t * 0.1, 5.0 - 0.05 * t, 9.0, 0.0, 0.0, 0.0] 
            for t in range(50)
        ]),
        'type': 'vehicle'
    }
    agents.append(agent2)
    
    return agents


def generate_obstacles() -> list:
    """生成障碍物"""
    obstacles = []
    
    # 静态障碍物1
    obstacle1 = {
        'x': 30.0,
        'y': 2.0,
        'radius': 2.0,
        'type': 'static'
    }
    obstacles.append(obstacle1)
    
    # 静态障碍物2
    obstacle2 = {
        'x': 40.0,
        'y': -1.5,
        'radius': 1.5,
        'type': 'static'
    }
    obstacles.append(obstacle2)
    
    return obstacles


def main():
    """主函数"""
    print("MARC规划器演示")
    print("=" * 50)
    
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'marc_config.json')
    config = load_config(config_path)
    
    # 创建规划器
    planner = MARCPlanner(config)
    
    # 生成场景数据
    initial_state = generate_initial_state()
    target_trajectory = generate_target_trajectory()
    exo_agents = generate_exo_agents()
    obstacles = generate_obstacles()
    
    print(f"初始状态: {initial_state}")
    print(f"规划范围: {len(target_trajectory)} 步")
    print(f"外部智能体数量: {len(exo_agents)}")
    print(f"障碍物数量: {len(obstacles)}")
    print()
    
    # 执行规划
    print("开始MARC规划...")
    result = planner.plan(initial_state, target_trajectory, exo_agents, obstacles)
    
    if result['success']:
        print("✓ 规划成功!")
        print(f"规划时间: {result['planning_time']:.3f} 秒")
        print(f"总成本: {result['cost']:.3f}")
        
        if 'trajectory' in result and result['trajectory'] is not None:
            print(f"轨迹长度: {len(result['trajectory'])}")
            print(f"控制序列长度: {len(result['controls'])}")
            
        if 'risk_metrics' in result:
            print("风险指标:")
            for key, value in result['risk_metrics'].items():
                print(f"  {key}: {value}")
                
        if 'branch_points' in result:
            print(f"分支点: {result['branch_points']}")
            
    else:
        print("✗ 规划失败!")
        print(f"原因: {result['reason']}")
        
    # 获取规划统计信息
    print("\n规划统计信息:")
    stats = planner.get_planning_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
        
    # 测试重新规划
    print("\n测试重新规划...")
    current_state = result['trajectory'][10] if result['success'] and 'trajectory' in result else initial_state
    replan_result = planner.replan(current_state, target_trajectory, exo_agents)
    
    if replan_result['success']:
        print("✓ 重新规划成功!")
        print(f"重新规划时间: {replan_result.get('planning_time', 0):.3f} 秒")
    else:
        print("✗ 重新规划失败!")
        print(f"原因: {replan_result['reason']}")
        
    print("\n演示完成!")


if __name__ == "__main__":
    main()