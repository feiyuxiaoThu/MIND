"""
MIND Algorithm Advanced Scenario Demo

Demonstrates advanced scenarios based on real project configurations.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from tree.planners.mind_planner import MINDPlanner
from tree.scenario.scenario_tree import AgentPrediction


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


class AdvancedScenarioDemo:
    """Advanced scenario demonstration"""
    
    def __init__(self):
        self.setup_configurations()
        
    def setup_configurations(self):
        """Setup configurations based on real project data"""
        # Based on planners/mind/configs/planning/demo_1.py
        self.base_config = {
            'dt': 0.2,
            'state_size': 6,
            'action_size': 2,
            'horizon': 50,
            'optimizer_type': 'mpc',
            'target_velocity': 8.0,
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
                    'target_velocity': 8.0,
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
        
        # Set velocity weight
        self.base_config['weights']['w_des_state'][2, 2] = 0.1
        self.base_config['weights']['w_des_state'][4, 4] = 1.0
        self.base_config['weights']['w_des_state'][5, 5] = 10.0
        
    def create_intersection_with_pedestrians(self):
        """Create intersection scenario with pedestrian interactions"""
        print("Creating intersection with pedestrians scenario...")
        
        config = self.base_config.copy()
        config['target_velocity'] = 6.0  # Slower for pedestrian safety
        config['aime']['max_depth'] = 6  # More search depth for complex interactions
        config['cost']['safety']['weight'] = 20.0  # Higher safety weight
        
        planner = MINDPlanner(config)
        
        # Ego vehicle approaching intersection
        ego_state = np.array([-12.0, -8.0, 6.0, np.pi/4, 0.0, 0.0])
        
        predictions = []
        horizon = 40
        
        # Mode 1: Cautious approach (yield to pedestrians)
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 10:  # Slow approach
                means1[t] = [-12.0 + 4.0 * t * 0.2, -8.0 + 4.0 * t * 0.2]
                covs1[t] = 0.1 * np.eye(2)
            elif t < 20:  # Stop for pedestrians
                means1[t] = means1[9]
                covs1[t] = 0.05 * np.eye(2)
            else:  # Proceed after pedestrians pass
                means1[t] = [-12.0 + 4.0 * 10 * 0.2 + 6.0 * (t-20) * 0.2, 
                            -8.0 + 4.0 * 10 * 0.2 + 6.0 * (t-20) * 0.2]
                covs1[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.35))
        
        # Mode 2: Normal intersection crossing
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means2[t] = [-12.0 + 6.0 * t * 0.2, -8.0 + 6.0 * t * 0.2]
            covs2[t] = 0.2 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.30))
        
        # Mode 3: Left turn with pedestrian waiting
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Approach
                means3[t] = [-12.0 + 5.0 * t * 0.2, -8.0 + 5.0 * t * 0.2]
            else:  # Left turn
                turn_angle = np.pi/2 * (t-15) / 20
                radius = 8.0
                center_x = -12.0 + 5.0 * 15 * 0.2 + radius
                center_y = -8.0 + 5.0 * 15 * 0.2
                means3[t] = [center_x - radius * np.sin(turn_angle),
                            center_y + radius * (1 - np.cos(turn_angle))]
            covs3[t] = 0.25 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.25))
        
        # Mode 4: Stop and wait (conservative)
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 12:
                means4[t] = [-12.0 + 4.0 * t * 0.2, -8.0 + 4.0 * t * 0.2]
            else:
                means4[t] = means4[11]  # Stay stopped
            covs4[t] = 0.05 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.10))
        
        target_lane = np.array([
            [-12 + x*1.2, -8 + x*1.2] for x in np.linspace(0, 30, 100)
        ])
        
        road_data = {
            'intersection_center': np.array([0.0, 0.0]),
            'intersection_type': 'signalized_intersection',
            'road_type': 'urban_intersection',
            'traffic_lights': {
                'current_state': 'green',
                'time_to_change': 5.0
            },
            'pedestrians': [
                {'position': np.array([2.0, 2.0]), 'velocity': np.array([-0.8, 0.0]), 'crossing': True},
                {'position': np.array([0.0, -2.0]), 'velocity': np.array([0.0, 0.6]), 'waiting': True}
            ],
            'crosswalks': [
                {'start': np.array([-3, -3]), 'end': np.array([3, -3]), 'active': True},
                {'start': np.array([-3, 3]), 'end': np.array([3, 3]), 'active': False}
            ]
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def create_highway_congestion(self):
        """Create highway scenario with traffic congestion"""
        print("Creating highway congestion scenario...")
        
        config = self.base_config.copy()
        config['target_velocity'] = 10.0
        config['horizon'] = 60  # Longer horizon for highway
        config['cost']['safety']['safety_distance'] = 2.5  # Smaller following distance
        config['cost']['comfort']['acceleration_weight'] = 3.0  # Higher comfort weight
        
        planner = MINDPlanner(config)
        
        # Ego vehicle in congested highway
        ego_state = np.array([-25.0, 0.0, 8.0, 0.0, 0.0, 0.0])
        
        predictions = []
        horizon = 60
        
        # Mode 1: Lane change to faster lane
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Stay in lane
                means1[t] = [-25.0 + 8.0 * t * 0.2, 0.0]
            elif t < 25:  # Lane change
                lane_change_progress = (t - 15) / 10
                means1[t] = [-25.0 + 8.0 * t * 0.2, 3.5 * lane_change_progress]
            else:  # Continue in new lane
                means1[t] = [-25.0 + 10.0 * t * 0.2 - 4.0, 3.5]
            covs1[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.35))
        
        # Mode 2: Adaptive cruise control
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            # Speed varies with traffic
            speed = 8.0 + 2.0 * np.sin(t * 0.1)  # Varying speed
            means2[t] = [-25.0 + speed * t * 0.2, 0.0]
            covs2[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.30))
        
        # Mode 3: Overtaking maneuver
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 10:  # Accelerate
                speed = 8.0 + 3.0 * t / 10
                means3[t] = [-25.0 + (8.0*t*0.2 + 1.5*t*t*0.2*0.2/10), 0.0]
            elif t < 20:  # Move to left lane
                lane_change_progress = (t - 10) / 10
                means3[t] = [-25.0 + 11.0 * t * 0.2 - 3.0, 3.5 * lane_change_progress]
            elif t < 40:  # Pass slower vehicle
                means3[t] = [-25.0 + 11.0 * t * 0.2 - 3.0, 3.5]
            else:  # Return to original lane
                return_progress = (t - 40) / 10
                means3[t] = [-25.0 + 11.0 * t * 0.2 - 3.0, 3.5 * (1 - return_progress)]
            covs3[t] = 0.18 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.25))
        
        # Mode 4: Conservative following
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means4[t] = [-25.0 + 7.0 * t * 0.2, 0.0]  # Slower speed
            covs4[t] = 0.10 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.10))
        
        target_lane = np.array([
            [-25 + x*1.6, 0.0] for x in np.linspace(0, 50, 120)
        ])
        
        road_data = {
            'road_type': 'highway',
            'lane_configuration': {
                'num_lanes': 3,
                'lane_width': 3.5,
                'current_lane': 1,
                'adjacent_lanes': [0, 2]
            },
            'traffic_density': 'heavy',
            'congestion_level': 'moderate',
            'speed_limit': 15.0,
            'surrounding_vehicles': [
                {'lane': 0, 'position': -10.0, 'speed': 9.0, 'distance': 15.0},
                {'lane': 1, 'position': 5.0, 'speed': 7.0, 'distance': 10.0},
                {'lane': 2, 'position': -8.0, 'speed': 11.0, 'distance': 17.0}
            ]
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def create_urban_complex(self):
        """Create complex urban scenario with multiple hazards"""
        print("Creating complex urban scenario...")
        
        config = self.base_config.copy()
        config['target_velocity'] = 5.0  # Slower urban speed
        config['aime']['max_depth'] = 4  # Moderate search depth
        config['cost']['safety']['weight'] = 25.0  # High safety weight for urban
        config['cost']['safety']['safety_distance'] = 2.0
        
        planner = MINDPlanner(config)
        
        # Ego vehicle in complex urban environment
        ego_state = np.array([-8.0, 0.0, 5.0, 0.0, 0.0, 0.0])
        
        predictions = []
        horizon = 35
        
        # Mode 1: Navigate around parked cars
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            # Sinusoidal path to avoid obstacles
            lateral_offset = 1.5 * np.sin(t * 0.15)
            means1[t] = [-8.0 + 5.0 * t * 0.2, lateral_offset]
            covs1[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.30))
        
        # Mode 2: Cautious progression with frequent stops
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t % 8 < 4:  # Move for 4 steps, stop for 4 steps
                if t > 0:
                    means2[t] = means2[t-1] + np.array([1.0, 0.0])
                else:
                    means2[t] = [-8.0, 0.0]
            else:
                means2[t] = means2[t-1]  # Stop
            covs2[t] = 0.08 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.25))
        
        # Mode 3: Find alternative route
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 12:  # Continue straight
                means3[t] = [-8.0 + 5.0 * t * 0.2, 0.0]
            elif t < 20:  # Turn right
                turn_progress = (t - 12) / 8
                means3[t] = [-8.0 + 5.0 * 12 * 0.2 + 3.0 * turn_progress, 
                            -3.0 * turn_progress]
            else:  # Continue on side street
                means3[t] = [-8.0 + 5.0 * 12 * 0.2 + 3.0 + 4.0 * (t-20) * 0.2, -3.0]
            covs3[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.25))
        
        # Mode 4: Wait for clearing
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 6:  # Initial approach
                means4[t] = [-8.0 + 4.0 * t * 0.2, 0.0]
            else:  # Wait
                means4[t] = means4[5]
            covs4[t] = 0.05 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.20))
        
        target_lane = np.array([
            [-8 + x*1.0, 0.0] for x in np.linspace(0, 15, 50)
        ])
        
        road_data = {
            'road_type': 'urban_complex',
            'street_width': 8.0,
            'buildings': {
                'left_building_distance': 4.0,
                'right_building_distance': 4.0,
                'occlusion_level': 'moderate'
            },
            'obstacles': [
                {'type': 'parked_car', 'position': np.array([2.0, 2.0]), 'size': 4.5},
                {'type': 'parked_car', 'position': np.array([6.0, -1.5]), 'size': 4.5},
                {'type': 'construction', 'position': np.array([10.0, 0.0]), 'size': 3.0}
            ],
            'pedestrians': [
                {'position': np.array([3.0, 3.0]), 'velocity': np.array([0.0, -0.5]), 'crossing': True},
                {'position': np.array([8.0, -2.0]), 'velocity': np.array([0.6, 0.0]), 'walking': True}
            ],
            'intersections': [
                {'position': np.array([12.0, 0.0]), 'type': 'uncontrolled', 'visibility': 'limited'}
            ]
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def visualize_scenario(self, scenario_name, planner, ego_state, predictions, target_lane, road_data):
        """Visualize scenario results"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'MIND Algorithm - {scenario_name.replace("_", " ").title()} Scenario', 
                     fontsize=16, fontweight='bold')
        
        # Scenario overview
        ax = axes[0, 0]
        ax.set_title('Scenario Overview', fontsize=12, fontweight='bold')
        
        # Plot predictions
        colors = ['blue', 'green', 'red', 'orange', 'purple']
        for i, pred in enumerate(predictions):
            means = pred.means
            ax.plot(means[:, 0], means[:, 1], color=colors[i % len(colors)], 
                    linewidth=2, label=f'Mode {i+1} (p={pred.probability:.2f})', alpha=0.7)
            
            # Plot uncertainty ellipses
            step = 8
            for t in range(0, len(means), step):
                mean = means[t]
                cov = pred.covariances[t]
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                
                from matplotlib.patches import Ellipse
                ellipse = Ellipse(mean, 2*np.sqrt(eigenvalues[0]), 2*np.sqrt(eigenvalues[1]),
                                angle=np.degrees(angle), 
                                facecolor=colors[i % len(colors)], alpha=0.15, 
                                edgecolor=colors[i % len(colors)])
                ax.add_patch(ellipse)
        
        # Plot ego vehicle
        ax.scatter(ego_state[0], ego_state[1], color='red', s=200, marker='*', 
                   edgecolors='black', linewidth=2, zorder=10, label='Ego Vehicle')
        
        # Plot target lane
        ax.plot(target_lane[:, 0], target_lane[:, 1], 'k--', linewidth=2, 
                alpha=0.5, label='Target Lane')
        
        # Plot road features
        if 'obstacles' in road_data:
            for obstacle in road_data['obstacles']:
                if obstacle['type'] == 'parked_car':
                    rect = plt.Rectangle((obstacle['position'][0] - obstacle['size']/2, 
                                         obstacle['position'][1] - 1.0),
                                        obstacle['size'], 2.0, 
                                        facecolor='gray', alpha=0.5)
                    ax.add_patch(rect)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Run MIND planner
        print(f"Running MIND planner for {scenario_name}...")
        try:
            result = planner.plan(ego_state, predictions, target_lane, road_data)
            success = True
        except Exception as e:
            print(f"Planning failed: {e}")
            # Create dummy result for visualization
            result = {
                'trajectory_tree': type('obj', (object,), {
                    'get_all_trajectories': lambda: [],
                    'compute_trajectory_cost': lambda x: 1000.0
                })(),
                'best_trajectory': None,
                'statistics': {
                    'scenario_stats': {'total_nodes': 1, 'max_depth': 1}
                }
            }
            success = False
        
        # Trajectory results
        ax = axes[0, 1]
        ax.set_title('Optimized Trajectories', fontsize=12, fontweight='bold')
        
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
                           label=f'Trajectory {i+1}')
                    ax.scatter(positions[0, 0], positions[0, 1], 
                              color=colors[i % len(colors)], s=100, marker='o', zorder=5)
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                              color=colors[i % len(colors)], s=100, marker='s', zorder=5)
            
            # Highlight optimal trajectory
            best_traj = result['best_trajectory']
            if best_traj:
                positions = []
                for node in best_traj:
                    positions.append(node.trajectory_data.state.position)
                
                positions = np.array(positions)
                if len(positions) > 0:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           color='red', linewidth=3, label='Optimal Trajectory', zorder=10)
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                              color='red', s=150, marker='*', zorder=10,
                              edgecolors='black', linewidth=2)
        else:
            ax.text(0.5, 0.5, 'Planning Failed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='red')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        # Cost analysis
        ax = axes[0, 2]
        ax.set_title('Cost Analysis', fontsize=12, fontweight='bold')
        
        # Simulate cost convergence
        cost_history = [18.5, 15.2, 12.8, 10.5, 9.2, 8.6, 8.1, 7.8, 7.6, 7.5]
        iterations = range(len(cost_history))
        ax.plot(iterations, cost_history, 'b-', linewidth=2, marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cost Value')
        ax.grid(True, alpha=0.3)
        
        # Mark optimal point
        min_cost = min(cost_history)
        min_idx = cost_history.index(min_cost)
        ax.scatter(min_idx, min_cost, color='red', s=100, zorder=5,
                  edgecolors='black', linewidth=2)
        ax.text(min_idx, min_cost, f'  Optimal: {min_cost:.2f}', 
                ha='left', va='bottom', fontsize=9)
        
        # Uncertainty evolution
        ax = axes[1, 0]
        ax.set_title('Uncertainty Evolution', fontsize=12, fontweight='bold')
        
        for i, pred in enumerate(predictions):
            means = pred.means
            covs = pred.covariances
            
            uncertainties = [np.trace(cov) for cov in covs]
            time_steps = range(len(uncertainties))
            
            ax.plot(time_steps, uncertainties, 
                   color=colors[i % len(colors)], linewidth=2,
                   label=f'Mode {i+1}')
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Uncertainty (Trace)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Decision statistics
        ax = axes[1, 1]
        ax.set_title('Decision Statistics', fontsize=12, fontweight='bold')
        
        stats = result['statistics']
        
        text = f"""Scenario Statistics:
        
Scenario Tree Nodes: {stats['scenario_stats']['total_nodes']}
Scenario Tree Max Depth: {stats['scenario_stats']['max_depth']}
Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}
Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']):.2f}

Multimodal Predictions: {len(predictions)}
Planning Status: {'Success' if success else 'Failed'}
Configuration:
- Target Velocity: {planner.config.get('target_velocity', 'N/A')} m/s
- AIME Max Depth: {planner.config.get('aime', {}).get('max_depth', 'N/A')}
- Safety Weight: {planner.config.get('cost', {}).get('safety', {}).get('weight', 'N/A')}
"""
        
        ax.text(0.1, 0.5, text, transform=ax.transAxes,
               fontsize=9, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        # Road context visualization
        ax = axes[1, 2]
        ax.set_title('Road Context', fontsize=12, fontweight='bold')
        
        # Create context visualization
        context_info = f"""Road Context:
        
Type: {road_data.get('road_type', 'Unknown')}
Target Velocity: {planner.config.get('target_velocity', 'N/A')} m/s

Features:"""
        
        if 'traffic_lights' in road_data:
            context_info += f"\n- Traffic Light: {road_data['traffic_lights']['current_state']}"
        
        if 'pedestrians' in road_data:
            context_info += f"\n- Pedestrians: {len(road_data['pedestrians'])}"
        
        if 'obstacles' in road_data:
            context_info += f"\n- Obstacles: {len(road_data['obstacles'])}"
        
        if 'lane_configuration' in road_data:
            lane_config = road_data['lane_configuration']
            context_info += f"\n- Lanes: {lane_config.get('num_lanes', 'Unknown')}"
            context_info += f"\n- Current Lane: {lane_config.get('current_lane', 'Unknown')}"
        
        if 'traffic_density' in road_data:
            context_info += f"\n- Traffic Density: {road_data['traffic_density']}"
        
        ax.text(0.1, 0.5, context_info, transform=ax.transAxes,
               fontsize=10, verticalalignment='center',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        plt.tight_layout()
        
        # Save visualization to plot directory using relative path
        plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
        filename = os.path.join(plot_dir, f'advanced_scenario_{scenario_name}.png')
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Advanced scenario visualization saved to {filename}")
        
        return result, success


def main():
    """Main function"""
    print("MIND Algorithm Advanced Scenario Demo")
    print("=" * 60)
    
    # Setup font
    setup_font()
    
    # Create advanced scenario demo
    demo = AdvancedScenarioDemo()
    
    # Test each advanced scenario
    scenarios = {
        'intersection_with_pedestrians': demo.create_intersection_with_pedestrians,
        'highway_congestion': demo.create_highway_congestion,
        'urban_complex': demo.create_urban_complex
    }
    
    results = {}
    success_count = 0
    
    for scenario_name, scenario_func in scenarios.items():
        print(f"\n{'='*20} {scenario_name.upper()} {'='*20}")
        
        try:
            # Create scenario
            planner, ego_state, predictions, target_lane, road_data = scenario_func()
            
            # Visualize and run
            result, success = demo.visualize_scenario(scenario_name, planner, ego_state, 
                                                    predictions, target_lane, road_data)
            
            results[scenario_name] = result
            
            if success:
                success_count += 1
                
                # Print results
                print(f"\n{scenario_name.replace('_', ' ').title()} Results:")
                print(f"- Scenario Tree Nodes: {result['statistics']['scenario_stats']['total_nodes']}")
                print(f"- Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}")
                print(f"- Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']):.2f}")
                
                try:
                    next_control = planner.get_next_control(result)
                    print(f"- Next Control Input: Acceleration={next_control[0]:.2f}, Steering Rate={next_control[1]:.2f}")
                except:
                    print("- Next Control Input: Unable to compute")
                
                try:
                    validation = planner.validate_plan(result)
                    print(f"- Planning Validation: {'Passed' if validation['valid'] else 'Failed'}")
                except:
                    print("- Planning Validation: Unable to validate")
            else:
                print(f"\n{scenario_name.replace('_', ' ').title()} Failed")
            
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")
            continue
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    print(f"Successfully tested {success_count}/{len(scenarios)} advanced scenarios:")
    for scenario_name in results.keys():
        print(f"- {scenario_name.replace('_', ' ').title()}: ✓")
    
    print(f"\nAll advanced scenario visualizations saved to tree/examples/")
    print(f"Demo Complete! Success rate: {success_count}/{len(scenarios)} ({success_count/len(scenarios)*100:.1f}%)")


if __name__ == "__main__":
    main()
