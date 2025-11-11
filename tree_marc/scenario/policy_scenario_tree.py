"""
基于策略条件的场景树生成器

实现MARC论文中的策略条件场景树生成算法，替代MIND的AIME算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PolicyData:
    """策略数据"""
    policy_type: str  # 策略类型
    probability: float
    target_lane: Optional[np.ndarray]
    metadata: Dict[str, Any]


@dataclass
class ScenarioData:
    """场景数据"""
    ego_trajectory: np.ndarray  # [T, state_dim]
    exo_trajectories: List[np.ndarray]  # [T, state_dim]
    probability: float
    policy_data: PolicyData
    branch_time: int
    end_time: int


class PolicyScenarioNode:
    """策略场景树节点"""
    
    def __init__(self, node_id: str, parent_id: Optional[str], 
                 scenario_data: ScenarioData, depth: int = 0):
        self.node_id = node_id
        self.parent_id = parent_id
        self.scenario_data = scenario_data
        self.depth = depth
        self.children_ids: List[str] = []
        
    @property
    def is_leaf(self) -> bool:
        return len(self.children_ids) == 0
        
    @property
    def is_root(self) -> bool:
        return self.parent_id is None


class PolicyScenarioTree:
    """策略场景树"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.nodes: Dict[str, PolicyScenarioNode] = {}
        
        # MARC参数
        self.max_depth = config.get('max_depth', 5)
        self.tar_dist_thres = config.get('tar_dist_thres', 10.0)
        self.tar_time_ahead = config.get('tar_time_ahead', 5.0)
        self.seg_length = config.get('seg_length', 15.0)
        self.seg_n_node = config.get('seg_n_node', 10)
        self.far_dist_thres = config.get('far_dist_thres', 10.0)
        
        # 分支参数
        self.divergence_threshold = config.get('divergence_threshold', 0.5)
        self.max_branch_time = config.get('max_branch_time', 20)
        self.probability_threshold = config.get('probability_threshold', 0.001)
        
    def reset(self):
        """重置场景树"""
        self.nodes = {}
        
    def add_root(self, scenario_data: ScenarioData) -> PolicyScenarioNode:
        """添加根节点"""
        root_node = PolicyScenarioNode("root", None, scenario_data, depth=0)
        self.nodes["root"] = root_node
        return root_node
        
    def add_scenario(self, parent_id: str, scenario_data: ScenarioData, 
                    node_id: Optional[str] = None) -> PolicyScenarioNode:
        """添加场景节点"""
        if node_id is None:
            node_id = f"{parent_id}_child_{len(self.nodes)}"
            
        scenario_node = PolicyScenarioNode(node_id, parent_id, scenario_data, depth=0)
        self.nodes[node_id] = scenario_node
        
        # 更新父节点的子节点列表
        if parent_id in self.nodes:
            self.nodes[parent_id].children_ids.append(node_id)
            
        return scenario_node
        
    def get_root(self) -> Optional[PolicyScenarioNode]:
        """获取根节点"""
        return self.nodes.get("root")
        
    def build_scenario_tree(self, scenarios: List[ScenarioData]) -> Dict[str, Any]:
        """
        构建场景树
        
        Args:
            scenarios: 场景数据列表
            
        Returns:
            scenario_tree: 场景树结构
        """
        # 创建根节点
        root_id = "root"
        root_policy_data = PolicyData(
            policy_type="root",
            probability=1.0,
            target_lane=np.array([0.0, 0.0, 1.0, 1.0]),
            metadata={}
        )
        root_scenario = ScenarioData(
            ego_trajectory=np.zeros((1, 6)),  # 简化的根轨迹
            exo_trajectories=[],
            probability=1.0,
            policy_data=root_policy_data,
            branch_time=0,
            end_time=0
        )
        self.add_scenario(root_id, root_scenario)
        
        # 添加场景节点
        for scenario in scenarios:
            self.add_scenario(root_id, scenario)
            
        return {
            'root_id': root_id,
            'num_scenarios': len(scenarios),
            'nodes': self.nodes
        }
        
    def get_leaf_nodes(self) -> List[PolicyScenarioNode]:
        """获取叶节点"""
        leaf_nodes = []
        for node in self.nodes.values():
            if node.is_leaf:
                leaf_nodes.append(node)
        return leaf_nodes
        
    def get_scenario_branches(self) -> List[List[PolicyScenarioNode]]:
        """获取从根到叶的所有场景分支"""
        branches = []
        leaf_nodes = self.get_leaf_nodes()
        
        for leaf_node in leaf_nodes:
            branch = []
            current_node = leaf_node
            while current_node is not None:
                branch.append(current_node)
                current_node = self.nodes.get(current_node.parent_id) if current_node.parent_id else None
            branches.append(list(reversed(branch)))
            
        return branches
        
    def generate_policy_conditioned_scenarios(self, ego_policies: List[str], 
                                            target_lane: np.ndarray,
                                            initial_state: np.ndarray) -> List[ScenarioData]:
        """生成策略条件场景"""
        scenarios = []
        
        for policy in ego_policies:
            # 为每个策略生成场景
            scenario_data = self._generate_policy_scenario(
                policy, target_lane, initial_state
            )
            scenarios.append(scenario_data)
            
        return scenarios
        
    def _generate_policy_scenario(self, policy: str, target_lane: np.ndarray,
                               initial_state: np.ndarray) -> ScenarioData:
        """生成单个策略场景"""
        horizon = 50
        dt = 0.1
        
        # 根据策略类型生成轨迹
        if policy == "lane_keeping":
            ego_trajectory = self._generate_lane_keeping_trajectory(
                initial_state, target_lane, horizon, dt
            )
        elif policy == "lane_change_left":
            ego_trajectory = self._generate_lane_change_trajectory(
                initial_state, target_lane, horizon, dt, direction="left"
            )
        elif policy == "lane_change_right":
            ego_trajectory = self._generate_lane_change_trajectory(
                initial_state, target_lane, horizon, dt, direction="right"
            )
        elif policy == "yielding":
            ego_trajectory = self._generate_yielding_trajectory(
                initial_state, target_lane, horizon, dt
            )
        else:
            # 默认直行轨迹
            ego_trajectory = self._generate_default_trajectory(
                initial_state, horizon, dt
            )
            
        # 生成外部智能体轨迹（简化处理）
        exo_trajectories = self._generate_exo_trajectories(
            ego_trajectory, horizon
        )
        
        # 创建策略数据
        policy_data = PolicyData(
            policy_type=policy,
            probability=1.0 / len(ego_policies) if hasattr(self, 'ego_policies') else 0.25,
            target_lane=target_lane,
            metadata={}
        )
        
        return ScenarioData(
            ego_trajectory=ego_trajectory,
            exo_trajectories=exo_trajectories,
            probability=policy_data.probability,
            policy_data=policy_data,
            branch_time=horizon,
            end_time=horizon
        )
        
    def _generate_lane_keeping_trajectory(self, initial_state: np.ndarray,
                                       target_lane: np.ndarray, horizon: int, dt: float) -> np.ndarray:
        """生成车道保持轨迹"""
        trajectory = np.zeros((horizon, 6))  # [x, y, v, theta, a, delta]
        
        x, y, v, theta, a, delta = initial_state
        
        for t in range(horizon):
            # 找到最近的目标点
            distances = [GeometryUtils.euclidean_distance([x, y], point) for point in target_lane]
            closest_idx = np.argmin(distances)
            
            # 设置目标状态
            target_pos = target_lane[closest_idx]
            target_velocity = 10.0  # 目标速度
            
            # 平滑调整到目标位置
            if t < 10:  # 前1秒平滑调整
                progress = t / 10
                target_x = x + (target_pos[0] - x) * progress * 0.1
                target_y = y + (target_pos[1] - y) * progress * 0.1
            else:
                target_x = target_pos[0]
                target_y = target_pos[1]
                
            # 计算目标航向
            if closest_idx < len(target_lane) - 1:
                next_pos = target_lane[closest_idx + 1]
                target_heading = np.arctan2(
                    next_pos[1] - target_pos[1],
                    next_pos[0] - target_pos[0]
                )
            else:
                target_heading = theta
                
            # 设置轨迹状态
            trajectory[t] = [target_x, target_y, target_velocity, target_heading, 0.0, 0.0]
            
            # 更新当前位置
            x = target_x
            y = target_y
            
        return trajectory
        
    def _generate_lane_change_trajectory(self, initial_state: np.ndarray,
                                      target_lane: np.ndarray, horizon: int, dt: float,
                                      direction: str) -> np.ndarray:
        """生成换道轨迹"""
        trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = initial_state
        lane_width = 3.5
        change_duration = 5.0  # 换道持续时间
        
        lateral_offset = lane_width if direction == "left" else -lane_width
        
        for t in range(horizon):
            if t < change_duration / dt:
                # 换道阶段
                progress = t / (change_duration / dt)
                lateral_progress = lateral_offset * (0.5 - 0.5 * np.cos(np.pi * progress))
                
                # 计算目标位置
                target_x = x + v * t * dt * np.cos(theta)
                target_y = y + v * t * dt * np.sin(theta) + lateral_progress
                
                # 调整航向
                target_heading = theta + np.arctan2(lateral_progress, v * t * dt)
            else:
                # 换道完成，保持车道
                target_x = x + v * t * dt * np.cos(theta)
                target_y = y + v * t * dt * np.sin(theta) + lateral_offset
                target_heading = theta
                
            trajectory[t] = [target_x, target_y, v, target_heading, 0.0, 0.0]
            
        return trajectory
        
    def _generate_yielding_trajectory(self, initial_state: np.ndarray,
                                     target_lane: np.ndarray, horizon: int, dt: float) -> np.ndarray:
        """生成让行轨迹"""
        trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = initial_state
        
        for t in range(horizon):
            # 让行轨迹：速度逐渐降低
            target_velocity = max(2.0, v * (1 - t / horizon))
            
            # 保持车道中心
            distances = [GeometryUtils.euclidean_distance([x, y], point) for point in target_lane]
            closest_idx = np.argmin(distances)
            target_pos = target_lane[closest_idx]
            
            trajectory[t] = [target_pos[0], target_pos[1], target_velocity, theta, 0.0, 0.0]
            
        return trajectory
        
    def _generate_default_trajectory(self, initial_state: np.ndarray,
                                  horizon: int, dt: float) -> np.ndarray:
        """生成默认直行轨迹"""
        trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = initial_state
        
        for t in range(horizon):
            trajectory[t] = [
                x + v * t * dt * np.cos(theta),
                y + v * t * dt * np.sin(theta),
                v, theta, 0.0, 0.0
            ]
            
        return trajectory
        
    def _generate_exo_trajectories(self, ego_trajectory: np.ndarray, 
                               horizon: int) -> List[np.ndarray]:
        """生成外部智能体轨迹"""
        exo_trajectories = []
        
        # 简化处理：生成3个外部车辆
        for i in range(3):
            exo_trajectory = np.zeros((horizon, 6))
            
            # 基于自车轨迹创建变体
            offset_angle = 2 * np.pi * i / 3
            offset_distance = 8.0
            
            for t in range(horizon):
                # 添加位置偏移
                offset_x = offset_distance * np.cos(offset_angle)
                offset_y = offset_distance * np.sin(offset_angle)
                
                exo_trajectory[t] = ego_trajectory[t].copy()
                exo_trajectory[t, 0] += offset_x
                exo_trajectory[t, 1] += offset_y
                
                # 添加速度变化
                speed_variation = 0.5 * np.sin(0.1 * t + i)
                exo_trajectory[t, 2] += speed_variation
                
            exo_trajectories.append(exo_trajectory)
            
        return exo_trajectories
        
    def build_tree_with_dynamic_branchpoints(self, scenarios: List[ScenarioData]) -> 'PolicyScenarioTree':
        """构建带有动态分支点的场景树"""
        self.reset()
        
        # 添加根节点（使用第一个场景）
        if scenarios:
            self.add_root(scenarios[0])
            
            # 为剩余场景添加节点
            for i, scenario in enumerate(scenarios[1:]):
                parent_id = "root"
                self.add_scenario(parent_id, scenario, f"scenario_{i}")
                
        # 计算动态分支点
        self._compute_dynamic_branchpoints()
        
        return self
        
    def _compute_dynamic_branchpoints(self):
        """计算动态分支点"""
        branches = self.get_scenario_branches()
        
        for branch in branches:
            if len(branch) < 2:
                continue
                
            # 计算分支时间
            branch_time = self._compute_branch_time(branch)
            
            # 更新所有节点的分支时间
            for node in branch:
                node.scenario_data.branch_time = branch_time
                
    def _compute_branch_time(self, branch: List[PolicyScenarioNode]) -> int:
        """计算分支时间"""
        if len(branch) < 2:
            return 0
            
        # 计算所有轨迹对之间的发散
        max_divergence_time = 0
        
        for i in range(len(branch)):
            for j in range(i + 1, len(branch)):
                divergence = self._compute_trajectory_divergence(
                    branch[i].scenario_data.ego_trajectory,
                    branch[j].scenario_data.ego_trajectory
                )
                
                # 找到最大发散时间
                for t, div in enumerate(divergence):
                    if div > self.divergence_threshold and t > max_divergence_time:
                        max_divergence_time = t
                        
        return min(max_divergence_time, self.max_branch_time)
        
    def _compute_trajectory_divergence(self, traj1: np.ndarray, traj2: np.ndarray) -> np.ndarray:
        """计算轨迹发散"""
        min_length = min(len(traj1), len(traj2))
        divergence = np.zeros(min_length)
        
        for t in range(min_length):
            # 计算位置差异
            pos_diff = np.linalg.norm(traj1[t, :2] - traj2[t, :2])
            divergence[t] = pos_diff
            
        return divergence