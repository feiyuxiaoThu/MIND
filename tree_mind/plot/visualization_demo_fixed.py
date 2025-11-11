"""
MIND Algorithm Visualization Demo - Font Fixed Version

Demonstrates algorithm performance at different iteration stages.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import sys
import os

# 添加项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction
from tree.scenarios.scenario_factory import ScenarioFactory


def setup_font():
    """Setup font for better display"""
    try:
        # Use default system fonts
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        print("Using default system fonts")
        return True
    except Exception as e:
        print(f"Font setup failed, using default: {e}")
        return False


class MindVisualizer:
    """MIND Algorithm Visualizer"""
    
    def __init__(self, figsize=(15, 10)):
        # Setup font first
        setup_font()
        
        self.fig, self.axes = plt.subplots(2, 3, figsize=figsize)
        self.fig.suptitle('MIND Algorithm Scenario Iteration Visualization', fontsize=16, fontweight='bold')
        
        # Set subplot titles
        titles = [
            'Scenario Tree Structure', 'Trajectory Optimization Results', 'Cost Convergence Curve',
            'Uncertainty Evolution', 'Multimodal Predictions', 'Decision Process'
        ]
        
        for i, ax in enumerate(self.axes.flat):
            ax.set_title(titles[i], fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
        self.scenario_tree_ax = self.axes[0, 0]
        self.trajectory_ax = self.axes[0, 1]
        self.cost_ax = self.axes[0, 2]
        self.uncertainty_ax = self.axes[1, 0]
        self.multimodal_ax = self.axes[1, 1]
        self.decision_ax = self.axes[1, 2]
        
        # 存储数据
        self.scenario_data = []
        self.trajectory_data = []
        self.cost_data = []
        self.uncertainty_data = []
        
    def create_intersection_scenario(self):
        """Create intersection scenario"""
        print("Creating intersection scenario...")
        
        config = {
            'dt': 0.2,
            'horizon': 30,
            'optimizer_type': 'mpc',
            'target_velocity': 8.0,
            'aime': {'max_depth': 4},
            'cost': {
                'safety': {'weight': 15.0, 'safety_distance': 3.0},
                'target': {'target_velocity': 8.0}
            }
        }
        
        planner = MINDPlanner(config)
        
        # 自车状态（接近路口）
        ego_state = np.array([-10.0, -5.0, 8.0, np.pi/4, 0.0, 0.0])
        
        # 创建多模态预测（路口场景）
        predictions = []
        horizon = 30
        
        # 模态1：直行通过
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means1[t] = [-10.0 + 8.0 * t * 0.2, -5.0 + 8.0 * t * 0.2]
            covs1[t] = 0.2 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.4))
        
        # 模态2：左转
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means2[t] = [-10.0 + 6.0 * t * 0.2, -5.0 + 6.0 * t * 0.2 - 0.5 * t * 0.2**2]
            covs2[t] = 0.3 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.3))
        
        # 模态3：右转
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means3[t] = [-10.0 + 6.0 * t * 0.2, -5.0 + 6.0 * t * 0.2 + 0.3 * t * 0.2**2]
            covs3[t] = 0.3 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.3))
        
        # 目标车道（直行）
        target_lane = np.array([[-10 + x, -5 + x] for x in np.linspace(0, 40, 100)])
        
        # 道路数据
        road_data = {
            'intersection_center': np.array([0.0, 0.0]),
            'intersection_type': 'four_way',
            'road_type': 'intersection'
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def visualize_scenario_tree(self, scenario_tree):
        """Visualize scenario tree structure"""
        self.scenario_tree_ax.clear()
        self.scenario_tree_ax.set_title('Scenario Tree Structure', fontsize=12, fontweight='bold')
        
        # 绘制树结构
        def draw_tree_recursive(node, x, y, level, width):
            if node.parent_key is None:
                # 根节点
                self.scenario_tree_ax.scatter(x, y, s=200, c='red', marker='o', 
                                           edgecolors='black', linewidth=2, zorder=5)
                self.scenario_tree_ax.text(x, y-1, 'Root', ha='center', fontsize=10, fontweight='bold')
            else:
                # 子节点
                self.scenario_tree_ax.scatter(x, y, s=150, c='lightblue', marker='o', 
                                           edgecolors='black', linewidth=1, zorder=4)
                self.scenario_tree_ax.text(x, y-1, f'{node.key[-1]}', ha='center', fontsize=9)
                
                # 连接线
                parent_x = x - level * width / (2**level)
                parent_y = y + 2
                self.scenario_tree_ax.plot([parent_x, x], [parent_y, y], 'k-', alpha=0.6, zorder=1)
            
            # 递归绘制子节点
            if hasattr(node, 'children_keys') and node.children_keys:
                num_children = len(node.children_keys)
                child_width = width / num_children
                start_x = x - width/2 + child_width/2
                
                for i, child_key in enumerate(node.children_keys):
                    child_node = scenario_tree.tree.get_node(child_key)
                    if child_node:
                        child_x = start_x + i * child_width
                        draw_tree_recursive(child_node, child_x, y-2, level+1, child_width)
            
        # 从根节点开始绘制
        root = scenario_tree.get_root()
        if root:
            draw_tree_recursive(root, 0, 0, 0, 8)
            
        self.scenario_tree_ax.set_xlim(-10, 10)
        self.scenario_tree_ax.set_ylim(-8, 2)
        self.scenario_tree_ax.grid(True, alpha=0.3)
        
    def visualize_trajectories(self, result):
        """Visualize trajectory optimization results"""
        self.trajectory_ax.clear()
        self.trajectory_ax.set_title('Trajectory Optimization Results', fontsize=12, fontweight='bold')
        
        # 绘制所有轨迹
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        
        trajectories = result['trajectory_tree'].get_all_trajectories()
        for i, traj in enumerate(trajectories):
            positions = []
            for node in traj:
                positions.append(node.trajectory_data.state.position)
            
            positions = np.array(positions)
            if len(positions) > 0:
                self.trajectory_ax.plot(positions[:, 0], positions[:, 1], 
                                      color=colors[i % len(colors)], alpha=0.6, linewidth=2,
                                      label=f'Trajectory {i+1}')
                self.trajectory_ax.scatter(positions[0, 0], positions[0, 1], 
                                           color=colors[i % len(colors)], s=100, marker='o', zorder=5)
                self.trajectory_ax.scatter(positions[-1, 0], positions[-1, 1], 
                                           color=colors[i % len(colors)], s=100, marker='s', zorder=5)
        
        # 高亮最优轨迹
        best_traj = result['best_trajectory']
        if best_traj:
            positions = []
            for node in best_traj:
                positions.append(node.trajectory_data.state.position)
            
            positions = np.array(positions)
            if len(positions) > 0:
                self.trajectory_ax.plot(positions[:, 0], positions[:, 1], 
                                      color='red', linewidth=3, label='Optimal Trajectory', zorder=10)
                self.trajectory_ax.scatter(positions[-1, 0], positions[-1, 1], 
                                           color='red', s=150, marker='*', zorder=10,
                                           edgecolors='black', linewidth=2)
        
        self.trajectory_ax.legend(loc='upper right', fontsize=8)
        self.trajectory_ax.grid(True, alpha=0.3)
        self.trajectory_ax.set_xlabel('X (m)', fontsize=10)
        self.trajectory_ax.set_ylabel('Y (m)', fontsize=10)
        
    def visualize_cost_convergence(self, cost_history):
        """Visualize cost convergence curve"""
        self.cost_ax.clear()
        self.cost_ax.set_title('Cost Convergence Curve', fontsize=12, fontweight='bold')
        
        if cost_history:
            iterations = range(len(cost_history))
            self.cost_ax.plot(iterations, cost_history, 'b-', linewidth=2, marker='o')
            self.cost_ax.set_xlabel('Iteration', fontsize=10)
            self.cost_ax.set_ylabel('Cost Value', fontsize=10)
            self.cost_ax.grid(True, alpha=0.3)
            
            # Mark convergence point
            if len(cost_history) > 1:
                min_cost = min(cost_history)
                min_idx = cost_history.index(min_cost)
                self.cost_ax.scatter(min_idx, min_cost, color='red', s=100, zorder=5,
                                    edgecolors='black', linewidth=2)
                self.cost_ax.text(min_idx, min_cost, f'  Optimal: {min_cost:.2f}', 
                                 ha='left', va='bottom', fontsize=9)
        
    def visualize_uncertainty_evolution(self, predictions):
        """Visualize uncertainty evolution"""
        self.uncertainty_ax.clear()
        self.uncertainty_ax.set_title('Uncertainty Evolution', fontsize=12, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        
        for i, pred in enumerate(predictions):
            means = pred.means
            covs = pred.covariances
            
            # 计算不确定性（迹）
            uncertainties = [np.trace(cov) for cov in covs]
            time_steps = range(len(uncertainties))
            
            self.uncertainty_ax.plot(time_steps, uncertainties, 
                                    color=colors[i % len(colors)], linewidth=2,
                                    label=f'Mode {i+1}')
            
            # 绘制置信区间
            means_array = np.array(means)
            std_dev = np.sqrt([np.trace(cov) for cov in covs])
            
            self.uncertainty_ax.fill_between(time_steps, 
                                          means_array[:, 1] - std_dev,
                                          means_array[:, 1] + std_dev,
                                          color=colors[i % len(colors)], alpha=0.2)
        
        self.uncertainty_ax.set_xlabel('Time Step', fontsize=10)
        self.uncertainty_ax.set_ylabel('Uncertainty (Trace)', fontsize=10)
        self.uncertainty_ax.legend(loc='upper right', fontsize=8)
        self.uncertainty_ax.grid(True, alpha=0.3)
        
    def visualize_multimodal_predictions(self, predictions):
        """Visualize multimodal predictions"""
        self.multimodal_ax.clear()
        self.multimodal_ax.set_title('Multimodal Predictions', fontsize=12, fontweight='bold')
        
        colors = ['blue', 'green', 'red']
        
        for i, pred in enumerate(predictions):
            means = pred.means
            covs = pred.covariances
            
            # 绘制均值轨迹
            self.multimodal_ax.plot(means[:, 0], means[:, 1], 
                                    color=colors[i % len(colors)], linewidth=2,
                                    label=f'Mode {i+1} (p={pred.probability:.2f})')
            
            # 绘制不确定性椭圆（每隔几步）
            step = 5
            for t in range(0, len(means), step):
                mean = means[t]
                cov = covs[t]
                
                # 计算椭圆参数
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                
                # 绘制椭圆
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(mean, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]),
                                angle=np.degrees(angle), 
                                facecolor=colors[i % len(colors)], alpha=0.2, 
                                edgecolor=colors[i % len(colors)])
                self.multimodal_ax.add_patch(ellipse)
        
        self.multimodal_ax.legend(loc='upper right', fontsize=8)
        self.multimodal_ax.grid(True, alpha=0.3)
        self.multimodal_ax.set_xlabel('X (m)', fontsize=10)
        self.multimodal_ax.set_ylabel('Y (m)', fontsize=10)
        
    def visualize_decision_process(self, result, predictions):
        """Visualize decision process"""
        self.decision_ax.clear()
        self.decision_ax.set_title('Decision Process', fontsize=12, fontweight='bold')
        
        # 显示决策统计
        stats = result['statistics']
        
        text = f"""Decision Statistics:
        
Scenario Tree Nodes: {stats['scenario_stats']['total_nodes']}
Scenario Tree Max Depth: {stats['scenario_stats']['max_depth']}
Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}
Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']):.2f}

Multimodal Predictions: {len(predictions)}
"""
        
        self.decision_ax.text(0.1, 0.5, text, transform=self.decision_ax.transAxes,
                            fontsize=10, verticalalignment='center',
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        self.decision_ax.set_xlim(0, 1)
        self.decision_ax.set_ylim(0, 1)
        self.decision_ax.axis('off')
        
    def create_animation(self, planner, ego_state, predictions, target_lane, road_data):
        """Create animation demonstration"""
        print("Creating animation demonstration...")
        
        # 运行规划并收集数据
        result = planner.plan(ego_state, predictions, target_lane, road_data)
        
        # 逐步可视化
        self.visualize_scenario_tree(result['scenario_tree'])
        self.visualize_trajectories(result)
        self.visualize_cost_convergence([10.5, 8.2, 7.8, 7.6, 7.5])  # 模拟成本收敛
        self.visualize_uncertainty_evolution(predictions)
        self.visualize_multimodal_predictions(predictions)
        self.visualize_decision_process(result, predictions)
        
        plt.tight_layout()
        return result
        
    def save_visualization(self, filename='mind_visualization_fixed.png'):
        """保存可视化结果"""
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {filename}")


def main():
    """Main function"""
    print("MIND Algorithm Visualization Demo - Font Fixed Version")
    print("=" * 50)
    
    # 创建可视化器
    visualizer = MindVisualizer()
    
    # 创建场景
    planner, ego_state, predictions, target_lane, road_data = visualizer.create_intersection_scenario()
    
    # 创建动画
    result = visualizer.create_animation(planner, ego_state, predictions, target_lane, road_data)
    
    # Save results
    visualizer.save_visualization('/home/feiyushaw/Documents/Work/e2e/MIND/tree/examples/mind_visualization_fixed.png')
    
    # Display result statistics
    print("\nAlgorithm Results:")
    print(f"- Scenario Tree Nodes: {result['statistics']['scenario_stats']['total_nodes']}")
    print(f"- Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}")
    print(f"- Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']):.2f}")
    
    # Get next control
    next_control = planner.get_next_control(result)
    print(f"- Next Control Input: Acceleration={next_control[0]:.2f}, Steering Rate={next_control[1]:.2f}")
    
    # Validate planning
    validation = planner.validate_plan(result)
    print(f"- Planning Validation: {'Passed' if validation['valid'] else 'Failed'}")
    
    print("\nDemo Complete!")
    print("Visualization saved, please check mind_visualization_fixed.png file")


if __name__ == "__main__":
    main()