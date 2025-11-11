"""
场景树核心实现

重构自原始MIND代码，去除深度学习依赖，支持多模态预测作为输入。
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from ..utils.tree import Tree, Node


@dataclass
class AgentState:
    """智能体状态"""
    position: np.ndarray  # [x, y]
    velocity: np.ndarray  # [vx, vy]
    heading: float        # 航向角
    timestamp: float      # 时间戳


@dataclass
class AgentPrediction:
    """智能体预测"""
    means: np.ndarray      # [T, 2] 位置均值
    covariances: np.ndarray # [T, 2, 2] 位置协方差
    probability: float     # 概率权重


@dataclass
class ScenarioData:
    """场景数据"""
    ego_prediction: AgentPrediction
    exo_predictions: List[AgentPrediction]
    probability: float
    timestamp: float
    metadata: Dict[str, Any]


class ScenarioNode(Node):
    """场景树节点"""
    
    def __init__(self, node_id: str, parent_id: Optional[str], 
                 scenario_data: ScenarioData, depth: int = 0):
        super().__init__(node_id, parent_id, scenario_data)
        self.depth = depth
        self.branch_time = None
        self.end_time = None
        self.is_branch_node = False
        self.is_end_node = False
        
    @property
    def scenario_data(self) -> ScenarioData:
        """获取场景数据"""
        return self.data
        
    @property
    def probability(self) -> float:
        """获取场景概率"""
        return self.scenario_data.probability


class ScenarioTree:
    """场景树类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tree = Tree()
        self.max_depth = config.get('max_depth', 5)
        self.obs_len = config.get('obs_len', 50)
        self.pred_len = config.get('pred_len', 60)
        
        # AIME参数
        self.uncertainty_threshold = config.get('uncertainty_threshold', 9.0)
        self.probability_threshold = config.get('probability_threshold', 0.001)
        self.topology_threshold = config.get('topology_threshold', np.pi / 6)
        
    def reset(self):
        """重置场景树"""
        self.tree = Tree()
        
    def add_root(self, scenario_data: ScenarioData):
        """添加根节点"""
        root_node = ScenarioNode("root", None, scenario_data, depth=0)
        self.tree.add_node(root_node)
        return root_node
        
    def add_scenario(self, parent_id: str, scenario_data: ScenarioData, 
                    node_id: Optional[str] = None) -> ScenarioNode:
        """添加场景节点"""
        if node_id is None:
            node_id = f"{parent_id}_child_{len(self.tree.get_node(parent_id).children_keys)}"
            
        parent_node = self.tree.get_node(parent_id)
        depth = parent_node.depth + 1
        
        scenario_node = ScenarioNode(node_id, parent_id, scenario_data, depth)
        self.tree.add_node(scenario_node)
        
        return scenario_node
        
    def get_root(self) -> ScenarioNode:
        """获取根节点"""
        return self.tree.get_root()
        
    def get_leaf_nodes(self) -> List[ScenarioNode]:
        """获取叶节点"""
        return self.tree.get_leaf_nodes()
        
    def get_branch_nodes(self) -> List[ScenarioNode]:
        """获取分支节点"""
        branch_nodes = []
        for node in self.get_leaf_nodes():
            if node.is_branch_node:
                branch_nodes.append(node)
        return branch_nodes
        
    def get_end_nodes(self) -> List[ScenarioNode]:
        """获取终端节点"""
        end_nodes = []
        for node in self.get_leaf_nodes():
            if node.is_end_node:
                end_nodes.append(node)
        return end_nodes
        
    def get_scenario_branches(self) -> List[List[ScenarioNode]]:
        """获取从根到叶的所有场景分支"""
        branches = []
        end_nodes = self.get_end_nodes()
        
        for end_node in end_nodes:
            branch = []
            current_node = end_node
            while current_node is not None:
                branch.append(current_node)
                current_node = self.tree.get_node(current_node.parent_key) if current_node.parent_key else None
            branches.append(list(reversed(branch)))
            
        return branches
        
    def calculate_branch_probability(self, branch: List[ScenarioNode]) -> float:
        """计算分支概率"""
        probability = 1.0
        for node in branch[1:]:  # 跳过根节点
            probability *= node.probability
        return probability
        
    def normalize_probabilities(self):
        """归一化所有分支的概率"""
        branches = self.get_scenario_branches()
        total_prob = sum(self.calculate_branch_probability(branch) for branch in branches)
        
        if total_prob > 0:
            for branch in branches:
                branch_prob = self.calculate_branch_probability(branch)
                normalized_prob = branch_prob / total_prob
                # 更新最后一个节点的概率作为分支概率
                branch[-1].scenario_data.probability = normalized_prob
                
    def prune_by_probability(self, threshold: float = None):
        """根据概率剪枝"""
        if threshold is None:
            threshold = self.probability_threshold
            
        nodes_to_remove = []
        for node in self.tree.nodes.values():
            if node.scenario_data.probability < threshold:
                nodes_to_remove.append(node.key)
                
        for node_id in nodes_to_remove:
            self.tree.remove_node(node_id)
            
    def get_max_depth(self) -> int:
        """获取树的最大深度"""
        max_depth = 0
        for node in self.tree.nodes.values():
            max_depth = max(max_depth, node.depth)
        return max_depth
        
    def get_statistics(self) -> Dict[str, Any]:
        """获取场景树统计信息"""
        return {
            'total_nodes': len(self.tree.nodes),
            'max_depth': self.get_max_depth(),
            'num_branches': len(self.get_branch_nodes()),
            'num_end_nodes': len(self.get_end_nodes()),
            'num_scenarios': len(self.get_scenario_branches())
        }