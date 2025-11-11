"""
轨迹树核心实现

重构自原始MIND代码，去除深度学习依赖，专注于轨迹优化和不确定性处理。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.tree import Tree, Node
from ..scenario.scenario_tree import ScenarioTree, ScenarioNode, ScenarioData


@dataclass
class TrajectoryState:
    """轨迹状态"""
    position: np.ndarray      # [x, y]
    velocity: float          # 速度大小
    heading: float           # 航向角
    acceleration: float      # 加速度
    steering_angle: float    # 转向角
    timestamp: float         # 时间戳


@dataclass
class ControlInput:
    """控制输入"""
    acceleration: float      # 加速度变化率
    steering_rate: float     # 转向角变化率


@dataclass
class TrajectoryData:
    """轨迹数据"""
    state: TrajectoryState
    control: ControlInput
    cost: float
    probability: float
    metadata: Dict[str, Any]


class TrajectoryNode(Node):
    """轨迹树节点"""
    
    def __init__(self, node_id: str, parent_id: Optional[str], 
                 trajectory_data: TrajectoryData, depth: int = 0):
        super().__init__(node_id, parent_id, trajectory_data)
        self.depth = depth
        self.time_index = 0
        
    @property
    def trajectory_data(self) -> TrajectoryData:
        """获取轨迹数据"""
        return self.data
        
    @property
    def state(self) -> TrajectoryState:
        """获取状态"""
        return self.trajectory_data.state
        
    @property
    def control(self) -> ControlInput:
        """获取控制输入"""
        return self.trajectory_data.control


class TrajectoryTree:
    """轨迹树类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tree = Tree()
        self.dt = config.get('dt', 0.1)
        self.horizon = config.get('horizon', 50)
        
    def reset(self):
        """重置轨迹树"""
        self.tree = Tree()
        
    def add_root(self, initial_state: TrajectoryState, initial_control: ControlInput):
        """添加根节点"""
        root_data = TrajectoryData(
            state=initial_state,
            control=initial_control,
            cost=0.0,
            probability=1.0,
            metadata={}
        )
        root_node = TrajectoryNode("root", None, root_data, depth=0)
        self.tree.add_node(root_node)
        return root_node
        
    def add_trajectory_step(self, parent_id: str, state: TrajectoryState, 
                          control: ControlInput, cost: float, 
                          probability: float = 1.0) -> TrajectoryNode:
        """添加轨迹步骤"""
        parent_node = self.tree.get_node(parent_id)
        depth = parent_node.depth + 1
        time_index = parent_node.time_index + 1
        
        trajectory_data = TrajectoryData(
            state=state,
            control=control,
            cost=cost,
            probability=probability,
            metadata={'time_index': time_index}
        )
        
        node_id = f"{parent_id}_t{time_index}"
        trajectory_node = TrajectoryNode(node_id, parent_id, trajectory_data, depth)
        trajectory_node.time_index = time_index
        
        self.tree.add_node(trajectory_node)
        return trajectory_node
        
    def get_trajectory_from_root(self, node_id: str) -> List[TrajectoryNode]:
        """获取从根节点到指定节点的轨迹"""
        path = []
        current_node = self.tree.get_node(node_id)
        
        while current_node is not None:
            path.append(current_node)
            current_node = self.tree.get_node(current_node.parent_key) if current_node.parent_key else None
            
        return list(reversed(path))
        
    def get_all_trajectories(self) -> List[List[TrajectoryNode]]:
        """获取所有完整轨迹"""
        trajectories = []
        leaf_nodes = self.tree.get_leaf_nodes()
        
        for leaf_node in leaf_nodes:
            trajectory = self.get_trajectory_from_root(leaf_node.key)
            trajectories.append(trajectory)
            
        return trajectories
        
    def compute_trajectory_cost(self, trajectory: List[TrajectoryNode]) -> float:
        """计算轨迹总成本"""
        total_cost = 0.0
        for node in trajectory[1:]:  # 跳过根节点
            total_cost += node.trajectory_data.cost
        return total_cost
        
    def compute_trajectory_probability(self, trajectory: List[TrajectoryNode]) -> float:
        """计算轨迹概率"""
        probability = 1.0
        for node in trajectory[1:]:  # 跳过根节点
            probability *= node.trajectory_data.probability
        return probability
        
    def get_best_trajectory(self) -> List[TrajectoryNode]:
        """获取最优轨迹（成本最低）"""
        trajectories = self.get_all_trajectories()
        if not trajectories:
            return []
            
        best_trajectory = min(trajectories, key=self.compute_trajectory_cost)
        return best_trajectory
        
    def get_probabilistic_best_trajectory(self) -> List[TrajectoryNode]:
        """获取概率最优轨迹（考虑成本和概率）"""
        trajectories = self.get_all_trajectories()
        if not trajectories:
            return []
            
        best_trajectory = None
        best_score = float('inf')
        
        for trajectory in trajectories:
            cost = self.compute_trajectory_cost(trajectory)
            probability = self.compute_trajectory_probability(trajectory)
            
            # 成本-概率权衡评分
            score = cost / (probability + 1e-6)
            
            if score < best_score:
                best_score = score
                best_trajectory = trajectory
                
        return best_trajectory
        
    def prune_high_cost_trajectories(self, cost_threshold: float):
        """剪枝高成本轨迹"""
        trajectories = self.get_all_trajectories()
        nodes_to_remove = []
        
        for trajectory in trajectories:
            cost = self.compute_trajectory_cost(trajectory)
            if cost > cost_threshold:
                # 移除整条轨迹
                for node in trajectory[1:]:  # 保留根节点
                    nodes_to_remove.append(node.key)
                    
        # 移除节点（从叶到根，避免破坏树结构）
        nodes_to_remove = list(set(nodes_to_remove))
        for node_id in nodes_to_remove:
            if node_id in self.tree.nodes:
                self.tree.remove_node(node_id)
                
    def get_trajectory_at_time(self, time_index: int) -> List[TrajectoryNode]:
        """获取指定时间步的所有节点"""
        nodes_at_time = []
        for node in self.tree.nodes.values():
            if node.time_index == time_index:
                nodes_at_time.append(node)
        return nodes_at_time
        
    def compute_state_statistics(self, time_index: int) -> Dict[str, Any]:
        """计算指定时间步的状态统计"""
        nodes_at_time = self.get_trajectory_at_time(time_index)
        if not nodes_at_time:
            return {}
            
        positions = np.array([node.state.position for node in nodes_at_time])
        velocities = np.array([node.state.velocity for node in nodes_at_time])
        headings = np.array([node.state.heading for node in nodes_at_time])
        
        return {
            'mean_position': np.mean(positions, axis=0),
            'position_std': np.std(positions, axis=0),
            'mean_velocity': np.mean(velocities),
            'velocity_std': np.std(velocities),
            'mean_heading': np.mean(headings),
            'heading_std': np.std(headings),
            'num_trajectories': len(nodes_at_time)
        }
        
    def extract_trajectory_arrays(self, trajectory: List[TrajectoryNode]) -> Dict[str, np.ndarray]:
        """提取轨迹数组用于分析"""
        if not trajectory:
            return {}
            
        positions = np.array([node.state.position for node in trajectory])
        velocities = np.array([node.state.velocity for node in trajectory])
        headings = np.array([node.state.heading for node in trajectory])
        accelerations = np.array([node.state.acceleration for node in trajectory])
        steering_angles = np.array([node.state.steering_angle for node in trajectory])
        
        controls = np.array([[node.control.acceleration, node.control.steering_rate] 
                           for node in trajectory[1:]])  # 根节点无控制
        
        costs = np.array([node.trajectory_data.cost for node in trajectory[1:]])
        
        return {
            'positions': positions,
            'velocities': velocities,
            'headings': headings,
            'accelerations': accelerations,
            'steering_angles': steering_angles,
            'controls': controls,
            'costs': costs
        }
        
    def validate_trajectory(self, trajectory: List[TrajectoryNode]) -> Dict[str, Any]:
        """验证轨迹的合理性"""
        if len(trajectory) < 2:
            return {'valid': False, 'reason': 'Trajectory too short'}
            
        issues = []
        
        # 检查状态连续性
        for i in range(1, len(trajectory)):
            prev_state = trajectory[i-1].state
            curr_state = trajectory[i].state
            
            # 检查位置跳跃
            position_diff = np.linalg.norm(curr_state.position - prev_state.position)
            max_reasonable_move = curr_state.velocity * self.dt * 2  # 允许2倍的理论位移
            
            if position_diff > max_reasonable_move:
                issues.append(f"Large position jump at step {i}: {position_diff:.2f}m")
                
            # 检查速度变化
            velocity_diff = abs(curr_state.velocity - prev_state.velocity)
            max_reasonable_accel = 5.0  # m/s²
            
            if velocity_diff / self.dt > max_reasonable_accel:
                issues.append(f"Large acceleration at step {i}: {velocity_diff/self.dt:.2f}m/s²")
                
            # 检查航向变化
            heading_diff = abs(curr_state.heading - prev_state.heading)
            heading_diff = min(heading_diff, 2*np.pi - heading_diff)
            max_reasonable_turn = np.pi/4  # 45度
            
            if heading_diff > max_reasonable_turn:
                issues.append(f"Large heading change at step {i}: {np.degrees(heading_diff):.1f}°")
                
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'num_issues': len(issues)
        }
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取轨迹树统计信息"""
        trajectories = self.get_all_trajectories()
        
        if not trajectories:
            return {
                'total_nodes': len(self.tree.nodes),
                'num_trajectories': 0,
                'max_depth': 0,
                'mean_cost': 0.0,
                'cost_std': 0.0
            }
            
        costs = [self.compute_trajectory_cost(traj) for traj in trajectories]
        depths = [len(traj) for traj in trajectories]
        
        return {
            'total_nodes': len(self.tree.nodes),
            'num_trajectories': len(trajectories),
            'max_depth': max(depths),
            'min_depth': min(depths),
            'mean_depth': np.mean(depths),
            'mean_cost': np.mean(costs),
            'cost_std': np.std(costs),
            'min_cost': min(costs),
            'max_cost': max(costs)
        }