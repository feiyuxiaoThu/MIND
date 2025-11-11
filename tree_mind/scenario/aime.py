"""
自适应交互模态探索(AIME)算法实现

重构自原始MIND代码，去除深度学习依赖。
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from .scenario_tree import ScenarioTree, ScenarioNode, ScenarioData, AgentPrediction
from ..utils.geometry import GeometryUtils


class AIME:
    """自适应交互模态探索算法"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # AIME参数
        self.max_depth = config.get('max_depth', 5)
        self.uncertainty_threshold = config.get('uncertainty_threshold', 9.0)
        self.probability_threshold = config.get('probability_threshold', 0.001)
        self.topology_threshold = config.get('topology_threshold', np.pi / 6)
        self.target_distance_threshold = config.get('target_distance_threshold', 10.0)
        
        # 分支参数
        self.min_branch_interval = config.get('min_branch_interval', 2)  # 最小分支间隔
        self.max_scenarios_per_branch = config.get('max_scenarios_per_branch', 6)
        
    def generate_scenario_tree(self, initial_predictions: List[ScenarioData], 
                             target_lane: np.ndarray = None) -> ScenarioTree:
        """生成场景树"""
        scenario_tree = ScenarioTree(self.config)
        
        # 初始化根节点
        if len(initial_predictions) > 0:
            root_data = initial_predictions[0]
            scenario_tree.add_root(root_data)
        else:
            raise ValueError("Initial predictions cannot be empty")
            
        # AIME迭代
        branch_nodes = [scenario_tree.get_root()]
        current_depth = 0
        
        while branch_nodes and current_depth < self.max_depth:
            new_branch_nodes = []
            
            for branch_node in branch_nodes:
                # 扩展场景
                expanded_scenarios = self._expand_scenarios(branch_node.scenario_data)
                
                # 剪枝和合并
                pruned_scenarios = self._prune_and_merge(expanded_scenarios, target_lane)
                
                # 创建新节点
                for scenario_data in pruned_scenarios:
                    new_node = scenario_tree.add_scenario(
                        branch_node.key, scenario_data
                    )
                    
                    # 决定是否继续分支
                    if self._should_branch(new_node):
                        new_node.is_branch_node = True
                        new_branch_nodes.append(new_node)
                    else:
                        new_node.is_end_node = True
                        
            branch_nodes = new_branch_nodes
            current_depth += 1
            
        # 标记剩余分支节点为终端节点
        for node in scenario_tree.get_leaf_nodes():
            if not node.is_end_node:
                node.is_end_node = True
                
        # 归一化概率
        scenario_tree.normalize_probabilities()
        
        return scenario_tree
        
    def _expand_scenarios(self, parent_scenario: ScenarioData) -> List[ScenarioData]:
        """扩展场景"""
        expanded_scenarios = []
        
        # 这里应该调用多模态预测模块
        # 在重构版本中，我们假设已经有多模态预测作为输入
        # 实际实现时需要根据具体的多模态预测方法进行调整
        
        # 示例：基于父场景生成多个子场景
        num_scenarios = np.random.randint(3, self.max_scenarios_per_branch + 1)
        
        for i in range(num_scenarios):
            # 创建变异的场景
            child_scenario = self._create_child_scenario(parent_scenario, i)
            expanded_scenarios.append(child_scenario)
            
        return expanded_scenarios
        
    def _create_child_scenario(self, parent_scenario: ScenarioData, 
                              scenario_idx: int) -> ScenarioData:
        """创建子场景"""
        # 基于父场景创建变异的场景
        ego_pred = parent_scenario.ego_prediction
        exo_preds = parent_scenario.exo_predictions
        
        # 添加随机扰动
        noise_scale = 0.1
        ego_means = ego_pred.means + np.random.normal(0, noise_scale, ego_pred.means.shape)
        ego_covs = ego_pred.covariances * (1 + np.random.normal(0, 0.1, ego_pred.covariances.shape))
        
        child_ego_pred = AgentPrediction(
            means=ego_means,
            covariances=ego_covs,
            probability=ego_pred.probability / max(len(exo_preds), 1)
        )
        
        # 对外部智能体也添加扰动
        child_exo_preds = []
        for exo_pred in exo_preds:
            exo_means = exo_pred.means + np.random.normal(0, noise_scale, exo_pred.means.shape)
            exo_covs = exo_pred.covariances * (1 + np.random.normal(0, 0.1, exo_pred.covariances.shape))
            
            child_exo_pred = AgentPrediction(
                means=exo_means,
                covariances=exo_covs,
                probability=exo_pred.probability / len(exo_preds)
            )
            child_exo_preds.append(child_exo_pred)
            
        return ScenarioData(
            ego_prediction=child_ego_pred,
            exo_predictions=child_exo_preds,
            probability=parent_scenario.probability / max(len(exo_preds), 1),
            timestamp=parent_scenario.timestamp,
            metadata=parent_scenario.metadata.copy()
        )
        
    def _prune_and_merge(self, scenarios: List[ScenarioData], 
                        target_lane: np.ndarray = None) -> List[ScenarioData]:
        """剪枝和合并场景"""
        # 1. 概率剪枝
        filtered_scenarios = []
        for scenario in scenarios:
            if scenario.probability >= self.probability_threshold:
                # 目标车道剪枝
                if target_lane is not None:
                    ego_final_pos = scenario.ego_prediction.means[-1]
                    distance = GeometryUtils.distance_to_polyline(ego_final_pos, target_lane)
                    if distance <= self.target_distance_threshold:
                        filtered_scenarios.append(scenario)
                else:
                    filtered_scenarios.append(scenario)
                    
        # 2. 拓扑合并
        merged_scenarios = self._merge_by_topology(filtered_scenarios)
        
        return merged_scenarios
        
    def _merge_by_topology(self, scenarios: List[ScenarioData]) -> List[ScenarioData]:
        """基于拓扑合并相似场景"""
        if len(scenarios) <= 1:
            return scenarios
            
        merged_scenarios = []
        used_indices = set()
        
        for i, scenario_i in enumerate(scenarios):
            if i in used_indices:
                continue
                
            # 找到具有相似拓扑的场景
            similar_scenarios = [scenario_i]
            used_indices.add(i)
            
            for j, scenario_j in enumerate(scenarios):
                if j <= i or j in used_indices:
                    continue
                    
                if self._are_topologically_similar(scenario_i, scenario_j):
                    similar_scenarios.append(scenario_j)
                    used_indices.add(j)
                    
            # 合并相似场景
            if len(similar_scenarios) > 1:
                merged_scenario = self._merge_scenarios(similar_scenarios)
                merged_scenarios.append(merged_scenario)
            else:
                merged_scenarios.append(scenario_i)
                
        return merged_scenarios
        
    def _are_topologically_similar(self, scenario1: ScenarioData, 
                                  scenario2: ScenarioData) -> bool:
        """判断两个场景是否拓扑相似"""
        # 计算交互模态
        interaction1 = self._compute_interaction_modality(scenario1)
        interaction2 = self._compute_interaction_modality(scenario2)
        
        # 计算拓扑差异
        topology_diff = np.abs(interaction1 - interaction2)
        topology_diff = GeometryUtils.normalize_angle(topology_diff)
        
        return np.abs(topology_diff) < self.topology_threshold
        
    def _compute_interaction_modality(self, scenario: ScenarioData) -> float:
        """计算交互模态"""
        ego_trajectory = scenario.ego_prediction.means
        interaction_modality = 0.0
        
        for exo_pred in scenario.exo_predictions:
            exo_trajectory = exo_pred.means
            
            # 计算自车到外部智能体的向量角度变化
            angle_changes = []
            for t in range(1, len(ego_trajectory)):
                ego_pos = ego_trajectory[t]
                exo_pos = exo_trajectory[t]
                
                # 计算向量
                vector = exo_pos - ego_pos
                if t > 1:
                    prev_vector = exo_trajectory[t-1] - ego_trajectory[t-1]
                    
                    # 计算角度变化
                    angle1 = np.arctan2(vector[1], vector[0])
                    angle2 = np.arctan2(prev_vector[1], prev_vector[0])
                    angle_diff = GeometryUtils.normalize_angle(angle1 - angle2)
                    angle_changes.append(angle_diff)
                    
            # 累积角度变化
            interaction_modality += np.sum(angle_changes)
            
        return interaction_modality
        
    def _merge_scenarios(self, scenarios: List[ScenarioData]) -> ScenarioData:
        """合并多个场景"""
        # 选择概率最高的场景作为基础
        base_scenario = max(scenarios, key=lambda s: s.probability)
        
        # 加权平均其他场景
        total_prob = sum(s.probability for s in scenarios)
        merged_prob = total_prob
        
        # 合并自车预测
        ego_means = np.average([s.ego_prediction.means for s in scenarios], 
                              weights=[s.probability for s in scenarios], axis=0)
        ego_covs = np.average([s.ego_prediction.covariances for s in scenarios], 
                             weights=[s.probability for s in scenarios], axis=0)
        
        merged_ego_pred = AgentPrediction(
            means=ego_means,
            covariances=ego_covs,
            probability=merged_prob
        )
        
        # 合并外部智能体预测
        merged_exo_preds = []
        if scenarios[0].exo_predictions:
            num_exo = len(scenarios[0].exo_predictions)
            for i in range(num_exo):
                exo_means = np.average([s.exo_predictions[i].means for s in scenarios], 
                                      weights=[s.probability for s in scenarios], axis=0)
                exo_covs = np.average([s.exo_predictions[i].covariances for s in scenarios], 
                                     weights=[s.probability for s in scenarios], axis=0)
                
                merged_exo_pred = AgentPrediction(
                    means=exo_means,
                    covariances=exo_covs,
                    probability=merged_prob / num_exo
                )
                merged_exo_preds.append(merged_exo_pred)
                
        return ScenarioData(
            ego_prediction=merged_ego_pred,
            exo_predictions=merged_exo_preds,
            probability=merged_prob,
            timestamp=scenarios[0].timestamp,
            metadata=scenarios[0].metadata.copy()
        )
        
    def _should_branch(self, node: ScenarioNode) -> bool:
        """决定是否应该继续分支"""
        # 检查深度限制
        if node.depth >= self.max_depth:
            return False
            
        # 检查不确定性变化
        scenario_data = node.scenario_data
        uncertainty_change = self._compute_uncertainty_change(scenario_data)
        
        if uncertainty_change < self.uncertainty_threshold:
            return False
            
        return True
        
    def _compute_uncertainty_change(self, scenario: ScenarioData) -> float:
        """计算不确定性变化率"""
        ego_covs = scenario.ego_prediction.covariances
        
        if len(ego_covs) < 2:
            return 0.0
            
        # 计算协方差变化率
        initial_cov = np.trace(ego_covs[0])
        final_cov = np.trace(ego_covs[-1])
        
        if initial_cov > 0:
            change_rate = final_cov / initial_cov
        else:
            change_rate = 1.0
            
        return change_rate