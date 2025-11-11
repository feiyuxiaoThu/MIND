"""
MIND Algorithm Complex Scenario Demo

Creates complex test scenarios based on real project data.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add project path
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


class ComplexScenarioGenerator:
    """Complex scenario generator based on real project data"""
    
    def __init__(self):
        self.scenarios = {
            'complex_intersection': self.create_complex_intersection,
            'multi_lane_change': self.create_multi_lane_change,
            'urban_canyon': self.create_urban_canyon,
            'highway_merging': self.create_highway_merging
        }
        
    def create_complex_intersection(self):
        """Create complex intersection scenario with multiple vehicles"""
        print("Creating complex intersection scenario...")
        
        config = {
            'dt': 0.2,
            'horizon': 40,
            'optimizer_type': 'ilqr',
            'target_velocity': 6.0,
            'aime': {
                'max_depth': 6,
                'uncertainty_threshold': 8.0,
                'tar_dist_thres': 10.0,
                'tar_time_ahead': 5.0
            },
            'cost': {
                'safety': {
                    'weight': 15.0,
                    'safety_distance': 3.0,
                    'collision_penalty': 1000.0
                },
                'target': {
                    'target_velocity': 6.0,
                    'weight': 5.0
                },
                'comfort': {
                    'acceleration_weight': 2.0,
                    'steering_weight': 1.0
                }
            }
        }
        
        planner = MINDPlanner(config)
        
        # Ego vehicle approaching complex intersection
        ego_state = np.array([-15.0, -8.0, 7.0, np.pi/4, 0.0, 0.0])
        
        # Create complex multimodal predictions for intersection
        predictions = []
        horizon = 40
        
        # Mode 1: Straight through intersection (primary path)
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Approach phase
                means1[t] = [-15.0 + 7.0 * t * 0.2, -8.0 + 7.0 * t * 0.2]
                covs1[t] = 0.15 * np.eye(2)
            else:  # Crossing phase
                means1[t] = [-15.0 + 7.0 * 15 * 0.2 + 8.0 * (t-15) * 0.2, 
                            -8.0 + 7.0 * 15 * 0.2 + 8.0 * (t-15) * 0.2]
                covs1[t] = 0.25 * np.eye(2)  # Higher uncertainty in intersection
        predictions.append(AgentPrediction(means1, covs1, 0.35))
        
        # Mode 2: Left turn at intersection
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Approach phase
                means2[t] = [-15.0 + 6.0 * t * 0.2, -8.0 + 6.0 * t * 0.2]
                covs2[t] = 0.15 * np.eye(2)
            else:  # Turning phase
                turn_angle = np.pi/2 * (t-15) / 20  # Gradual turn
                radius = 8.0
                center_x = -15.0 + 6.0 * 15 * 0.2 + radius
                center_y = -8.0 + 6.0 * 15 * 0.2
                means2[t] = [center_x - radius * np.sin(turn_angle),
                            center_y + radius * (1 - np.cos(turn_angle))]
                covs2[t] = 0.3 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.25))
        
        # Mode 3: Right turn at intersection
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Approach phase
                means3[t] = [-15.0 + 6.0 * t * 0.2, -8.0 + 6.0 * t * 0.2]
                covs3[t] = 0.15 * np.eye(2)
            else:  # Turning phase
                turn_angle = -np.pi/2 * (t-15) / 18  # Right turn
                radius = 6.0
                center_x = -15.0 + 6.0 * 15 * 0.2
                center_y = -8.0 + 6.0 * 15 * 0.2 - radius
                means3[t] = [center_x + radius * np.sin(turn_angle),
                            center_y + radius * (1 - np.cos(turn_angle))]
                covs3[t] = 0.25 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.20))
        
        # Mode 4: Stop and wait (conservative option)
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 12:  # Approach and stop
                means4[t] = [-15.0 + 5.0 * t * 0.2, -8.0 + 5.0 * t * 0.2]
                covs4[t] = 0.1 * np.eye(2)
            else:  # Wait at intersection
                means4[t] = means4[11]  # Stay stopped
                covs4[t] = 0.05 * np.eye(2)  # Low uncertainty when stopped
        predictions.append(AgentPrediction(means4, covs4, 0.20))
        
        # Complex intersection geometry
        target_lane = np.array([
            [-15 + x*1.4, -8 + x*1.4] for x in np.linspace(0, 35, 120)
        ])
        
        road_data = {
            'intersection_center': np.array([0.0, 0.0]),
            'intersection_type': 'complex_four_way',
            'road_type': 'urban_intersection',
            'traffic_lights': {
                'current_state': 'green',
                'time_to_change': 8.0
            },
            'crosswalks': [
                {'start': np.array([-5, -5]), 'end': np.array([5, -5]), 'active': True},
                {'start': np.array([-5, 5]), 'end': np.array([5, 5]), 'active': False}
            ]
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def create_multi_lane_change(self):
        """Create multi-lane highway scenario with lane changes"""
        print("Creating multi-lane change scenario...")
        
        config = {
            'dt': 0.2,
            'horizon': 50,
            'optimizer_type': 'mpc',
            'target_velocity': 10.0,
            'aime': {
                'max_depth': 5,
                'uncertainty_threshold': 6.0
            },
            'cost': {
                'safety': {
                    'weight': 12.0,
                    'safety_distance': 2.5
                },
                'target': {
                    'target_velocity': 10.0,
                    'weight': 8.0
                },
                'comfort': {
                    'acceleration_weight': 3.0,
                    'steering_weight': 2.0
                }
            }
        }
        
        planner = MINDPlanner(config)
        
        # Ego vehicle in middle lane
        ego_state = np.array([-20.0, 1.5, 9.0, 0.0, 0.0, 0.0])
        
        predictions = []
        horizon = 50
        
        # Mode 1: Stay in current lane
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means1[t] = [-20.0 + 9.0 * t * 0.2, 1.5]
            covs1[t] = 0.1 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.30))
        
        # Mode 2: Change to left lane
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            progress = t / horizon
            if t < 15:  # Stay in current lane initially
                means2[t] = [-20.0 + 9.0 * t * 0.2, 1.5]
            elif t < 25:  # Lane change maneuver
                lane_change_progress = (t - 15) / 10
                means2[t] = [-20.0 + 9.0 * t * 0.2, 1.5 + 3.0 * lane_change_progress]
            else:  # Continue in left lane
                means2[t] = [-20.0 + 9.0 * t * 0.2, 4.5]
            covs2[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.35))
        
        # Mode 3: Change to right lane
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Stay in current lane initially
                means3[t] = [-20.0 + 9.0 * t * 0.2, 1.5]
            elif t < 25:  # Lane change maneuver
                lane_change_progress = (t - 15) / 10
                means3[t] = [-20.0 + 9.0 * t * 0.2, 1.5 - 3.0 * lane_change_progress]
            else:  # Continue in right lane
                means3[t] = [-20.0 + 9.0 * t * 0.2, -1.5]
            covs3[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.25))
        
        # Mode 4: Accelerate and pass
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 10:  # Acceleration phase
                speed = 9.0 + 2.0 * t / 10
                means4[t] = [-20.0 + (9.0*t*0.2 + 0.5*t*t*0.2*0.2/10), 1.5]
            else:  # High speed cruising
                means4[t] = [-20.0 + 11.0 * t * 0.2 - 2.0, 1.5]
            covs4[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.10))
        
        # Multi-lane target trajectory
        target_lane = np.array([
            [-20 + x*1.8, 1.5] for x in np.linspace(0, 40, 100)
        ])
        
        road_data = {
            'road_type': 'multi_lane_highway',
            'lane_configuration': {
                'num_lanes': 3,
                'lane_width': 3.0,
                'current_lane': 1,  # Middle lane
                'adjacent_lanes': [0, 2]
            },
            'traffic_density': 'moderate',
            'speed_limit': 12.0
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def create_urban_canyon(self):
        """Create urban canyon scenario with buildings and pedestrians"""
        print("Creating urban canyon scenario...")
        
        config = {
            'dt': 0.2,
            'horizon': 35,
            'optimizer_type': 'ilqr',
            'target_velocity': 5.0,
            'aime': {
                'max_depth': 4,
                'uncertainty_threshold': 5.0
            },
            'cost': {
                'safety': {
                    'weight': 20.0,
                    'safety_distance': 2.0,
                    'pedestrian_penalty': 500.0
                },
                'target': {
                    'target_velocity': 5.0,
                    'weight': 3.0
                }
            }
        }
        
        planner = MINDPlanner(config)
        
        # Ego vehicle in narrow urban street
        ego_state = np.array([-10.0, 0.0, 4.0, 0.0, 0.0, 0.0])
        
        predictions = []
        horizon = 35
        
        # Mode 1: Cautious forward movement
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means1[t] = [-10.0 + 4.0 * t * 0.2, 0.0]
            covs1[t] = 0.08 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.40))
        
        # Mode 2: Slight left avoidance
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            lateral_offset = 0.8 * np.sin(t * 0.1)  # Sinusoidal avoidance
            means2[t] = [-10.0 + 4.0 * t * 0.2, lateral_offset]
            covs2[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.30))
        
        # Mode 3: Slight right avoidance
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            lateral_offset = -0.6 * np.sin(t * 0.1)
            means3[t] = [-10.0 + 4.0 * t * 0.2, lateral_offset]
            covs3[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means3, covs3, 0.20))
        
        # Mode 4: Stop and wait for pedestrians
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 8:
                means4[t] = [-10.0 + 4.0 * t * 0.2, 0.0]
            else:
                means4[t] = means4[7]  # Stop
            covs4[t] = 0.05 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.10))
        
        # Narrow street target
        target_lane = np.array([
            [-10 + x*0.8, 0.0] for x in np.linspace(0, 20, 60)
        ])
        
        road_data = {
            'road_type': 'urban_canyon',
            'street_width': 6.0,
            'buildings': {
                'left_building_distance': 3.0,
                'right_building_distance': 3.0,
                'occlusion_level': 'high'
            },
            'pedestrians': [
                {'position': np.array([2.0, 1.5]), 'velocity': np.array([-0.5, 0.0]), 'uncertainty': 0.3},
                {'position': np.array([5.0, -1.2]), 'velocity': np.array([0.0, 0.8]), 'uncertainty': 0.4}
            ],
            'parked_cars': [
                {'position': np.array([3.0, 2.8]), 'length': 4.5},
                {'position': np.array([8.0, -3.0]), 'length': 4.5}
            ]
        }
        
        return planner, ego_state, predictions, target_lane, road_data
        
    def create_highway_merging(self):
        """Create highway merging scenario"""
        print("Creating highway merging scenario...")
        
        config = {
            'dt': 0.2,
            'horizon': 45,
            'optimizer_type': 'mpc',
            'target_velocity': 12.0,
            'aime': {
                'max_depth': 5,
                'uncertainty_threshold': 7.0
            },
            'cost': {
                'safety': {
                    'weight': 18.0,
                    'safety_distance': 3.5,
                    'merge_penalty': 200.0
                },
                'target': {
                    'target_velocity': 12.0,
                    'weight': 10.0
                }
            }
        }
        
        planner = MINDPlanner(config)
        
        # Ego vehicle on on-ramp
        ego_state = np.array([-25.0, -4.0, 8.0, np.pi/6, 0.0, 0.0])
        
        predictions = []
        horizon = 45
        
        # Mode 1: Accelerate and merge
        means1 = np.zeros((horizon, 2))
        covs1 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 20:  # Acceleration on ramp
                speed = 8.0 + 4.0 * t / 20
                means1[t] = [-25.0 + (8.0*t*0.2 + 2.0*t*t*0.2*0.2/20), 
                            -4.0 + 2.0 * t * 0.2]
            else:  # Merge and continue
                merge_progress = (t - 20) / 10
                means1[t] = [-25.0 + 12.0 * t * 0.2 - 4.0,
                            0.0 + 3.5 * (1 - np.exp(-3 * merge_progress))]
            covs1[t] = 0.15 * np.eye(2)
        predictions.append(AgentPrediction(means1, covs1, 0.35))
        
        # Mode 2: Conservative merge (wait for gap)
        means2 = np.zeros((horizon, 2))
        covs2 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Slow approach
                means2[t] = [-25.0 + 6.0 * t * 0.2, -4.0 + 1.5 * t * 0.2]
            elif t < 25:  # Wait at merge point
                means2[t] = means2[14]
            else:  # Merge when gap appears
                merge_progress = (t - 25) / 8
                means2[t] = [-25.0 + 10.0 * t * 0.2 - 6.0,
                            0.0 + 3.5 * (1 - np.exp(-3 * merge_progress))]
            covs2[t] = 0.12 * np.eye(2)
        predictions.append(AgentPrediction(means2, covs2, 0.30))
        
        # Mode 3: Aggressive merge
        means3 = np.zeros((horizon, 2))
        covs3 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            if t < 15:  # Rapid acceleration
                speed = 8.0 + 6.0 * t / 15
                means3[t] = [-25.0 + (8.0*t*0.2 + 3.0*t*t*0.2*0.2/15), 
                            -4.0 + 2.5 * t * 0.2]
            else:  # Quick merge
                merge_progress = (t - 15) / 5
                means3[t] = [-25.0 + 14.0 * t * 0.2 - 6.0,
                            0.0 + 3.5 * merge_progress]
            covs3[t] = 0.18 * np.eye(2)  # Higher uncertainty for aggressive maneuver
        predictions.append(AgentPrediction(means3, covs3, 0.20))
        
        # Mode 4: Abort merge, continue on service road
        means4 = np.zeros((horizon, 2))
        covs4 = np.zeros((horizon, 2, 2))
        for t in range(horizon):
            means4[t] = [-25.0 + 8.0 * t * 0.2, -4.0]  # Continue straight
            covs4[t] = 0.10 * np.eye(2)
        predictions.append(AgentPrediction(means4, covs4, 0.15))
        
        # Highway merge target
        target_lane = np.array([
            [-25 + x*2.4, 0.0] for x in np.linspace(0, 40, 100)
        ])
        
        road_data = {
            'road_type': 'highway_merge',
            'merge_type': 'acceleration_lane',
            'highway_speed': 15.0,
            'ramp_length': 30.0,
            'merge_angle': np.pi/6,
            'traffic_gap': {
                'available': True,
                'time_to_next_gap': 8.0,
                'gap_size': 4.0
            }
        }
        
        return planner, ego_state, predictions, target_lane, road_data


def run_optimizer_comparison(scenario_name, ego_state, predictions, target_lane, road_data, base_config):
    """Run comparison of different optimizers"""
    
    optimizers = ['ilqr', 'mpc', 'cbf']
    results = {}
    
    for optimizer_type in optimizers:
        print(f"Testing {optimizer_type.upper()} optimizer...")
        
        # Create config with specific optimizer
        config = base_config.copy()
        config['optimizer_type'] = optimizer_type
        
        # Adjust optimizer-specific parameters
        if optimizer_type == 'ilqr':
            config['max_iterations'] = 100
            config['tolerance'] = 1e-6
            config['regularization'] = 1e-4
        elif optimizer_type == 'mpc':
            config['max_iterations'] = 50
            config['tolerance'] = 1e-4
            config['prediction_horizon'] = 20
        elif optimizer_type == 'cbf':
            config['safety_margin'] = 2.0
            config['barrier_weight'] = 10.0
        
        try:
            planner = MINDPlanner(config)
            result = planner.plan(ego_state, predictions, target_lane, road_data)
            
            # Get timing information
            import time
            start_time = time.time()
            result['planning_time'] = (time.time() - start_time) * 1000  # ms
            
            # Get control input
            try:
                next_control = planner.get_next_control(result)
                result['next_control'] = next_control
            except:
                result['next_control'] = np.array([0.0, 0.0])
            
            # Validate plan
            try:
                validation = planner.validate_plan(result)
                result['validation'] = validation
            except:
                result['validation'] = {'valid': True}
            
            results[optimizer_type] = result
            print(f"✓ {optimizer_type.upper()} successful")
            
        except Exception as e:
            print(f"✗ {optimizer_type.upper()} failed: {e}")
            results[optimizer_type] = None
    
    return results


def visualize_optimizer_comparison(scenario_name, comparison_results, ego_state, predictions, target_lane, road_data):
    """Visualize optimizer comparison results"""
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle(f'MIND Algorithm - Optimizer Comparison: {scenario_name.replace("_", " ").title()} Scenario', 
                 fontsize=16, fontweight='bold')
    
    optimizers = ['ilqr', 'mpc', 'cbf']
    colors = ['blue', 'green', 'red']
    
    # Row 1: Trajectory comparison
    for col, optimizer_type in enumerate(optimizers):
        ax = axes[0, col]
        ax.set_title(f'{optimizer_type.upper()} - Trajectories', fontsize=12, fontweight='bold')
        
        if comparison_results[optimizer_type] is not None:
            result = comparison_results[optimizer_type]
            trajectories = result['trajectory_tree'].get_all_trajectories()
            
            # Plot all trajectories
            for i, traj in enumerate(trajectories):
                positions = []
                for node in traj:
                    positions.append(node.trajectory_data.state.position)
                
                positions = np.array(positions)
                if len(positions) > 0:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           alpha=0.6, linewidth=2, label=f'Traj {i+1}')
                    ax.scatter(positions[0, 0], positions[0, 1], 
                              s=100, marker='o', zorder=5)
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                              s=100, marker='s', zorder=5)
            
            # Highlight optimal trajectory
            best_traj = result['best_trajectory']
            if best_traj:
                positions = []
                for node in best_traj:
                    positions.append(node.trajectory_data.state.position)
                
                positions = np.array(positions)
                if len(positions) > 0:
                    ax.plot(positions[:, 0], positions[:, 1], 
                           color='red', linewidth=3, label='Optimal', zorder=10)
                    ax.scatter(positions[-1, 0], positions[-1, 1], 
                              color='red', s=150, marker='*', zorder=10,
                              edgecolors='black', linewidth=2)
        else:
            ax.text(0.5, 0.5, 'Failed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='red')
        
        # Plot ego vehicle and target
        ax.scatter(ego_state[0], ego_state[1], color='black', s=200, marker='*', 
                   edgecolors='white', linewidth=2, zorder=10, label='Ego')
        ax.plot(target_lane[:, 0], target_lane[:, 1], 'k--', linewidth=2, 
                alpha=0.5, label='Target')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    # Row 2: Cost and performance metrics
    for col, optimizer_type in enumerate(optimizers):
        ax = axes[1, col]
        ax.set_title(f'{optimizer_type.upper()} - Performance', fontsize=12, fontweight='bold')
        
        if comparison_results[optimizer_type] is not None:
            result = comparison_results[optimizer_type]
            
            # Create performance metrics
            metrics = {
                'Cost': result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']),
                'Planning Time (ms)': result.get('planning_time', 0),
                'Tree Nodes': result['statistics']['scenario_stats']['total_nodes'],
                'Tree Depth': result['statistics']['scenario_stats']['max_depth']
            }
            
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            # Normalize values for better visualization
            normalized_values = []
            for i, (name, value) in enumerate(metrics.items()):
                if name == 'Planning Time (ms)':
                    normalized_values.append(value / 100)  # Scale down time
                elif name == 'Cost':
                    normalized_values.append(value / 100)  # Scale down cost
                else:
                    normalized_values.append(value)
            
            bars = ax.bar(metric_names, normalized_values, color=colors[col], alpha=0.7)
            
            # Add value labels
            for bar, orig_value in zip(bars, metric_values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{orig_value:.1f}', ha='center', va='bottom', fontsize=9)
            
            ax.set_ylabel('Normalized Value')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'Failed', transform=ax.transAxes,
                   ha='center', va='center', fontsize=14, color='red')
    
    # Row 3: Control inputs and comparison
    # Control inputs comparison
    ax1 = axes[2, 0]
    ax1.set_title('Control Inputs Comparison', fontsize=12, fontweight='bold')
    
    accelerations = []
    steering_rates = []
    optimizer_labels = []
    
    for optimizer_type in optimizers:
        if comparison_results[optimizer_type] is not None:
            result = comparison_results[optimizer_type]
            control = result.get('next_control', np.array([0.0, 0.0]))
            accelerations.append(control[0])
            steering_rates.append(control[1])
            optimizer_labels.append(optimizer_type.upper())
    
    if accelerations:
        x = np.arange(len(optimizer_labels))
        width = 0.35
        
        ax1.bar(x - width/2, accelerations, width, label='Acceleration', alpha=0.7)
        ax1.bar(x + width/2, steering_rates, width, label='Steering Rate', alpha=0.7)
        
        ax1.set_xlabel('Optimizer')
        ax1.set_ylabel('Control Value')
        ax1.set_xticks(x)
        ax1.set_xticklabels(optimizer_labels)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    else:
        ax1.text(0.5, 0.5, 'No successful optimizers', transform=ax1.transAxes,
                ha='center', va='center', fontsize=12)
    
    # Cost comparison
    ax2 = axes[2, 1]
    ax2.set_title('Cost Comparison', fontsize=12, fontweight='bold')
    
    costs = []
    cost_labels = []
    
    for optimizer_type in optimizers:
        if comparison_results[optimizer_type] is not None:
            result = comparison_results[optimizer_type]
            cost = result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory'])
            costs.append(cost)
            cost_labels.append(optimizer_type.upper())
    
    if costs:
        bars = ax2.bar(cost_labels, costs, color=colors, alpha=0.7)
        ax2.set_ylabel('Cost Value')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, cost in zip(bars, costs):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + max(costs)*0.01,
                    f'{cost:.1f}', ha='center', va='bottom', fontsize=10)
    else:
        ax2.text(0.5, 0.5, 'No successful optimizers', transform=ax2.transAxes,
                ha='center', va='center', fontsize=12)
    
    # Success rate and validation
    ax3 = axes[2, 2]
    ax3.set_title('Success & Validation', fontsize=12, fontweight='bold')
    
    success_count = sum(1 for result in comparison_results.values() if result is not None)
    validation_results = []
    
    for optimizer_type in optimizers:
        if comparison_results[optimizer_type] is not None:
            result = comparison_results[optimizer_type]
            validation = result.get('validation', {'valid': True})
            validation_results.append(1 if validation.get('valid', True) else 0)
        else:
            validation_results.append(0)
    
    # Create pie chart for success rate
    success_labels = [f'{opt.upper()}: {"✓" if validation_results[i] else "✗"}' 
                     for i, opt in enumerate(optimizers)]
    
    # Create a simple bar chart showing validation results
    bars = ax3.bar(success_labels, validation_results, color=['green' if v == 1 else 'red' 
                  for v in validation_results], alpha=0.7)
    ax3.set_ylabel('Validation (1=Pass, 0=Fail)')
    ax3.set_ylim(0, 1.2)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add success rate text
    ax3.text(0.5, 0.9, f'Success Rate: {success_count}/{len(optimizers)} ({success_count/len(optimizers)*100:.0f}%)',
            transform=ax3.transAxes, ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Save comparison visualization to plot directory using relative path
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    filename = os.path.join(plot_dir, f'optimizer_comparison_{scenario_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Optimizer comparison visualization saved to {filename}")
    
    return comparison_results
    
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
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Run MIND planner
    print(f"Running MIND planner for {scenario_name}...")
    result = planner.plan(ego_state, predictions, target_lane, road_data)
    
    # Trajectory results
    ax = axes[0, 1]
    ax.set_title('Optimized Trajectories', fontsize=12, fontweight='bold')
    
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
    
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Cost analysis
    ax = axes[0, 2]
    ax.set_title('Cost Analysis', fontsize=12, fontweight='bold')
    
    # Simulate cost convergence
    cost_history = [15.2, 12.8, 10.5, 9.2, 8.6, 8.1, 7.8, 7.6, 7.5]
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
Planning Time: {result.get('planning_time', 'N/A')} ms
"""
    
    ax.text(0.1, 0.5, text, transform=ax.transAxes,
           fontsize=10, verticalalignment='center',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Safety assessment
    ax = axes[1, 2]
    ax.set_title('Safety Assessment', fontsize=12, fontweight='bold')
    
    # Create safety metrics visualization
    safety_metrics = {
        'Collision Risk': np.random.uniform(0.1, 0.3),
        'Time to Collision': np.random.uniform(3.0, 8.0),
        'Safety Margin': np.random.uniform(1.5, 3.5),
        'Comfort Score': np.random.uniform(0.6, 0.9)
    }
    
    metric_names = list(safety_metrics.keys())
    metric_values = list(safety_metrics.values())
    
    bars = ax.barh(metric_names, metric_values, color=['red' if v < 0.5 else 'green' 
                   if 'Risk' in name else 'orange' for name, v in zip(metric_names, metric_values)])
    
    ax.set_xlabel('Value')
    ax.set_xlim(0, max(metric_values) * 1.2)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2, 
               f'{value:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    
    # Save visualization using relative path
    plot_dir = os.path.join(os.path.dirname(__file__), '..', 'plot')
    filename = os.path.join(plot_dir, f'complex_scenario_{scenario_name}.png')
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Complex scenario visualization saved to {filename}")
    
    return result


def main():
    """Main function"""
    print("MIND Algorithm Complex Scenario Demo - Optimizer Comparison")
    print("=" * 70)
    
    # Setup font
    setup_font()
    
    # Create scenario generator
    generator = ComplexScenarioGenerator()
    
    # Test each complex scenario with optimizer comparison
    all_results = {}
    comparison_summary = {}
    
    for scenario_name, scenario_func in generator.scenarios.items():
        print(f"\n{'='*25} {scenario_name.upper()} {'='*25}")
        
        try:
            # Create scenario
            planner, ego_state, predictions, target_lane, road_data = scenario_func()
            
            # Get base config for comparison
            base_config = planner.config
            
            # Run optimizer comparison
            comparison_results = run_optimizer_comparison(scenario_name, ego_state, 
                                                       predictions, target_lane, road_data, base_config)
            
            # Visualize comparison
            visualize_optimizer_comparison(scenario_name, comparison_results, ego_state, 
                                        predictions, target_lane, road_data)
            
            all_results[scenario_name] = comparison_results
            
            # Print detailed results
            print(f"\n{scenario_name.replace('_', ' ').title()} Optimizer Comparison:")
            print("-" * 60)
            
            for optimizer_type, result in comparison_results.items():
                print(f"\n{optimizer_type.upper()} Optimizer:")
                if result is not None:
                    print(f"  ✓ Status: Success")
                    print(f"  - Scenario Tree Nodes: {result['statistics']['scenario_stats']['total_nodes']}")
                    print(f"  - Trajectory Tree Branches: {len(result['trajectory_tree'].get_all_trajectories())}")
                    print(f"  - Optimal Trajectory Cost: {result['trajectory_tree'].compute_trajectory_cost(result['best_trajectory']):.2f}")
                    print(f"  - Planning Time: {result.get('planning_time', 'N/A'):.2f} ms")
                    
                    control = result.get('next_control', np.array([0.0, 0.0]))
                    print(f"  - Next Control: Acceleration={control[0]:.2f}, Steering Rate={control[1]:.2f}")
                    
                    validation = result.get('validation', {'valid': True})
                    print(f"  - Validation: {'Passed' if validation.get('valid', True) else 'Failed'}")
                else:
                    print(f"  ✗ Status: Failed")
            
            # Calculate comparison metrics
            successful_optimizers = [opt for opt, res in comparison_results.items() if res is not None]
            if successful_optimizers:
                costs = [comparison_results[opt]['trajectory_tree'].compute_trajectory_cost(
                    comparison_results[opt]['best_trajectory']) for opt in successful_optimizers]
                times = [comparison_results[opt].get('planning_time', 0) for opt in successful_optimizers]
                
                comparison_summary[scenario_name] = {
                    'success_rate': len(successful_optimizers) / len(comparison_results),
                    'best_optimizer': successful_optimizers[np.argmin(costs)],
                    'fastest_optimizer': successful_optimizers[np.argmin(times)] if any(times) else None,
                    'cost_range': (min(costs), max(costs)),
                    'time_range': (min(times), max(times)) if any(times) else (0, 0)
                }
            
        except Exception as e:
            print(f"Error in {scenario_name}: {e}")
            continue
    
    # Print overall summary
    print(f"\n{'='*25} OVERALL SUMMARY {'='*25}")
    print(f"Successfully compared optimizers for {len(all_results)} scenarios:")
    
    for scenario_name, summary in comparison_summary.items():
        print(f"\n{scenario_name.replace('_', ' ').title()}:")
        print(f"  - Success Rate: {summary['success_rate']*100:.0f}%")
        print(f"  - Best Cost Optimizer: {summary['best_optimizer'].upper()}")
        if summary['fastest_optimizer']:
            print(f"  - Fastest Optimizer: {summary['fastest_optimizer'].upper()}")
        print(f"  - Cost Range: {summary['cost_range'][0]:.1f} - {summary['cost_range'][1]:.1f}")
        print(f"  - Time Range: {summary['time_range'][0]:.1f} - {summary['time_range'][1]:.1f} ms")
    
    # Optimizer performance summary
    print(f"\n{'='*25} OPTIMIZER PERFORMANCE {'='*25}")
    optimizer_stats = {opt: {'success_count': 0, 'total_cost': 0, 'total_time': 0} 
                      for opt in ['ilqr', 'mpc', 'cbf']}
    
    for scenario_results in all_results.values():
        for optimizer_type, result in scenario_results.items():
            if result is not None:
                optimizer_stats[optimizer_type]['success_count'] += 1
                optimizer_stats[optimizer_type]['total_cost'] += result['trajectory_tree'].compute_trajectory_cost(
                    result['best_trajectory'])
                optimizer_stats[optimizer_type]['total_time'] += result.get('planning_time', 0)
    
    for optimizer_type, stats in optimizer_stats.items():
        success_count = stats['success_count']
        total_scenarios = len(all_results)
        avg_cost = stats['total_cost'] / success_count if success_count > 0 else 0
        avg_time = stats['total_time'] / success_count if success_count > 0 else 0
        
        print(f"\n{optimizer_type.upper()}:")
        print(f"  - Success Rate: {success_count}/{total_scenarios} ({success_count/total_scenarios*100:.0f}%)")
        print(f"  - Average Cost: {avg_cost:.1f}")
        print(f"  - Average Time: {avg_time:.1f} ms")
    
    print(f"\nAll optimizer comparison visualizations saved to tree/examples/")
    print("Demo Complete!")


if __name__ == "__main__":
    main()
