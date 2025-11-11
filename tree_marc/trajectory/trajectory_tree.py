"""
MARC轨迹树

实现MARC论文中的轨迹树结构和优化算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from ..planning.cvar_optimizer import CVAROptimizer
from ..planning.bilevel_optimization import BilevelOptimization


@dataclass
class TrajectoryNode:
    """轨迹节点"""
    node_id: str
    state: np.ndarray
    control: np.ndarray
    parent_id: Optional[str] = None
    children_ids: List[str] = None
    cost_to_come: float = 0.0
    cost_to_go: float = 0.0
    probability: float = 1.0
    scenario_id: Optional[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
            
    @property
    def total_cost(self) -> float:
        """总成本"""
        return self.cost_to_come + self.cost_to_go


class SimpleGraph:
    """简单的图结构，替代networkx"""
    
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        
    def add_node(self, node_id: str, node: Any):
        """添加节点"""
        self.nodes[node_id] = {'node': node}
        
    def add_edge(self, source: str, target: str):
        """添加边"""
        if source not in self.edges:
            self.edges[source] = []
        self.edges[source].append(target)
        
    def successors(self, node_id: str) -> List[str]:
        """获取后继节点"""
        return self.edges.get(node_id, [])
        
    def number_of_nodes(self) -> int:
        """节点数量"""
        return len(self.nodes)
        
    def number_of_edges(self) -> int:
        """边数量"""
        return sum(len(targets) for targets in self.edges.values())
        
    def remove_node(self, node_id: str):
        """移除节点"""
        if node_id in self.nodes:
            del self.nodes[node_id]
        if node_id in self.edges:
            del self.edges[node_id]
        # 移除指向该节点的边
        for source in self.edges:
            if node_id in self.edges[source]:
                self.edges[source].remove(node_id)


class MARCTrajectoryTree:
    """MARC轨迹树"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 轨迹树参数
        self.max_depth = config.get('max_depth', 10)
        self.branching_factor = config.get('branching_factor', 3)
        self.prune_threshold = config.get('prune_threshold', 1000.0)
        self.min_probability = config.get('min_probability', 0.01)
        
        # 优化器
        self.cvar_optimizer = CVAROptimizer(config)
        self.bilevel_optimization = BilevelOptimization(config)
        
        # 轨迹树结构
        self.graph = SimpleGraph()
        self.root_node = None
        self.leaf_nodes = []
        self.scenario_branches = {}
        
    def initialize_tree(self, initial_state: np.ndarray, 
                       target_trajectory: np.ndarray) -> str:
        """
        初始化轨迹树
        
        Args:
            initial_state: 初始状态
            target_trajectory: 目标轨迹
            
        Returns:
            root_id: 根节点ID
        """
        # 创建根节点
        root_id = "root"
        root_node = TrajectoryNode(
            node_id=root_id,
            state=initial_state,
            control=np.zeros(2),
            cost_to_come=0.0,
            cost_to_go=self._compute_cost_to_go(initial_state, target_trajectory),
            probability=1.0
        )
        
        # 添加到图中
        self.graph.add_node(root_id, root_node)
        self.root_node = root_node
        self.leaf_nodes = [root_id]
        
        return root_id
        
    def expand_tree(self, scenarios: List[Dict[str, Any]], 
                   target_trajectory: np.ndarray,
                   branch_points: List[int]) -> Dict[str, Any]:
        """
        扩展轨迹树
        
        Args:
            scenarios: 场景列表
            target_trajectory: 目标轨迹
            branch_points: 分支点时间步列表
            
        Returns:
            expansion_result: 扩展结果
        """
        if not self.root_node:
            return {
                'success': False,
                'reason': 'Tree not initialized'
            }
            
        expansion_result = {
            'success': True,
            'added_nodes': 0,
            'pruned_nodes': 0,
            'scenarios_processed': 0
        }
        
        # 处理每个场景
        for scenario in scenarios:
            scenario_id = scenario.get('id', f'scenario_{len(self.scenario_branches)}')
            
            # 为场景创建分支
            branch_result = self._create_scenario_branches(
                scenario, target_trajectory, branch_points
            )
            
            if branch_result['success']:
                self.scenario_branches[scenario_id] = branch_result['branch_nodes']
                expansion_result['added_nodes'] += branch_result['nodes_added']
                expansion_result['scenarios_processed'] += 1
                
        # 剪枝操作
        prune_result = self._prune_tree()
        expansion_result['pruned_nodes'] = prune_result['nodes_pruned']
        
        return expansion_result
        
    def _create_scenario_branches(self, scenario: Dict[str, Any], 
                                 target_trajectory: np.ndarray,
                                 branch_points: List[int]) -> Dict[str, Any]:
        """
        为场景创建分支
        
        Args:
            scenario: 场景数据
            target_trajectory: 目标轨迹
            branch_points: 分支点列表
            
        Returns:
            branch_result: 分支创建结果
        """
        scenario_id = scenario.get('id', 'unknown')
        probability = scenario.get('probability', 1.0)
        
        branch_nodes = []
        nodes_added = 0
        
        # 当前叶节点
        current_leaf = self.root_node.node_id
        current_state = self.root_node.state
        
        # 沿时间步扩展
        for t, target_state in enumerate(target_trajectory):
            # 检查是否是分支点
            if t in branch_points and t > 0:
                # 创建多个分支
                branch_result = self._create_branches_at_time(
                    current_leaf, current_state, target_state, 
                    scenario, t, probability
                )
                
                if branch_result['success']:
                    branch_nodes.extend(branch_result['new_nodes'])
                    nodes_added += branch_result['nodes_added']
                    
                    # 选择最优分支继续扩展
                    current_leaf = self._select_best_branch(branch_result['new_nodes'])
                    current_state = self.graph.nodes[current_leaf]['node'].state
            else:
                # 单一扩展
                extend_result = self._extend_single_node(
                    current_leaf, current_state, target_state, scenario, t
                )
                
                if extend_result['success']:
                    current_leaf = extend_result['new_node_id']
                    current_state = self.graph.nodes[current_leaf]['node'].state
                    branch_nodes.append(current_leaf)
                    nodes_added += 1
                    
        return {
            'success': True,
            'branch_nodes': branch_nodes,
            'nodes_added': nodes_added
        }
        
    def _create_branches_at_time(self, parent_id: str, current_state: np.ndarray,
                                target_state: np.ndarray, scenario: Dict[str, Any],
                                time_step: int, probability: float) -> Dict[str, Any]:
        """
        在指定时间步创建分支
        
        Args:
            parent_id: 父节点ID
            current_state: 当前状态
            target_state: 目标状态
            scenario: 场景数据
            time_step: 时间步
            probability: 概率
            
        Returns:
            branch_result: 分支创建结果
        """
        new_nodes = []
        
        # 生成多个候选控制策略
        candidate_controls = self._generate_candidate_controls(
            current_state, target_state, scenario
        )
        
        for i, control in enumerate(candidate_controls):
            # 计算下一状态
            next_state = self._simulate_step(current_state, control)
            
            # 计算成本
            cost_to_come = self.graph.nodes[parent_id]['node'].cost_to_come + \
                          self._compute_step_cost(current_state, next_state, control)
            cost_to_go = self._compute_cost_to_go(next_state, \
                                                 np.array([target_state] * max(1, self.max_depth - time_step)))
            
            # 创建节点
            node_id = f"{parent_id}_t{time_step}_b{i}"
            node = TrajectoryNode(
                node_id=node_id,
                state=next_state,
                control=control,
                parent_id=parent_id,
                cost_to_come=cost_to_come,
                cost_to_go=cost_to_go,
                probability=probability / len(candidate_controls),
                scenario_id=scenario.get('id')
            )
            
            # 添加到图中
            self.graph.add_node(node_id, node)
            self.graph.add_edge(parent_id, node_id)
            
            new_nodes.append(node_id)
            
        return {
            'success': True,
            'new_nodes': new_nodes,
            'nodes_added': len(new_nodes)
        }
        
    def _extend_single_node(self, parent_id: str, current_state: np.ndarray,
                           target_state: np.ndarray, scenario: Dict[str, Any],
                           time_step: int) -> Dict[str, Any]:
        """
        单一节点扩展
        
        Args:
            parent_id: 父节点ID
            current_state: 当前状态
            target_state: 目标状态
            scenario: 场景数据
            time_step: 时间步
            
        Returns:
            extend_result: 扩展结果
        """
        # 生成最优控制
        optimal_control = self._generate_optimal_control(
            current_state, target_state, scenario
        )
        
        # 计算下一状态
        next_state = self._simulate_step(current_state, optimal_control)
        
        # 计算成本
        parent_node = self.graph.nodes[parent_id]['node']
        cost_to_come = parent_node.cost_to_come + \
                      self._compute_step_cost(current_state, next_state, optimal_control)
        cost_to_go = self._compute_cost_to_go(next_state, \
                                             np.array([target_state] * max(1, self.max_depth - time_step)))
        
        # 创建节点
        node_id = f"{parent_id}_t{time_step}"
        node = TrajectoryNode(
            node_id=node_id,
            state=next_state,
            control=optimal_control,
            parent_id=parent_id,
            cost_to_come=cost_to_come,
            cost_to_go=cost_to_go,
            probability=parent_node.probability,
            scenario_id=scenario.get('id')
        )
        
        # 添加到图中
        self.graph.add_node(node_id, node)
        self.graph.add_edge(parent_id, node_id)
        
        return {
            'success': True,
            'new_node_id': node_id
        }
        
    def _generate_candidate_controls(self, current_state: np.ndarray, 
                                   target_state: np.ndarray,
                                   scenario: Dict[str, Any]) -> List[np.ndarray]:
        """生成候选控制策略"""
        candidates = []
        
        # 基础控制：朝向目标
        dx = target_state[0] - current_state[0]
        dy = target_state[1] - current_state[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance > 0.1:
            # 朝向目标的方向
            target_angle = np.arctan2(dy, dx)
            current_angle = current_state[3]
            angle_diff = self._normalize_angle(target_angle - current_angle)
            
            # 基础控制
            base_control = np.array([
                np.clip(distance * 0.5, -2.0, 2.0),  # 加速度
                np.clip(angle_diff * 2.0, -0.3, 0.3)  # 转向角
            ])
            candidates.append(base_control)
            
            # 保守控制
            conservative_control = base_control * 0.7
            candidates.append(conservative_control)
            
            # 激进控制
            aggressive_control = base_control * 1.3
            aggressive_control[0] = np.clip(aggressive_control[0], -3.0, 3.0)
            aggressive_control[1] = np.clip(aggressive_control[1], -0.5, 0.5)
            candidates.append(aggressive_control)
            
        else:
            # 保持当前控制
            candidates.append(np.array([0.0, 0.0]))
            
        return candidates[:self.branching_factor]
        
    def _generate_optimal_control(self, current_state: np.ndarray,
                                 target_state: np.ndarray,
                                 scenario: Dict[str, Any]) -> np.ndarray:
        """生成最优控制"""
        # 简化版本：返回朝向目标的控制
        candidates = self._generate_candidate_controls(current_state, target_state, scenario)
        
        if not candidates:
            return np.array([0.0, 0.0])
            
        # 评估每个候选控制
        best_control = candidates[0]
        best_cost = float('inf')
        
        for control in candidates:
            next_state = self._simulate_step(current_state, control)
            cost = self._compute_step_cost(current_state, next_state, control)
            
            if cost < best_cost:
                best_cost = cost
                best_control = control
                
        return best_control
        
    def _simulate_step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """模拟单步动力学"""
        x, y, v, theta, a, delta = state
        da, ddelta = control
        
        # 更新状态
        a_next = np.clip(a + da * 0.1, -3.0, 3.0)
        delta_next = np.clip(delta + ddelta * 0.1, -0.5, 0.5)
        
        v_next = v + a_next * 0.1
        v_next = max(0.0, v_next)
        
        x_next = x + v_next * np.cos(theta) * 0.1
        y_next = y + v_next * np.sin(theta) * 0.1
        theta_next = theta + v_next / 2.5 * np.tan(delta_next) * 0.1
        
        return np.array([x_next, y_next, v_next, theta_next, a_next, delta_next])
        
    def _compute_step_cost(self, current_state: np.ndarray, next_state: np.ndarray,
                          control: np.ndarray) -> float:
        """计算单步成本"""
        cost = 0.0
        
        # 控制成本
        cost += np.sum(control**2) * 0.1
        
        # 状态变化成本
        state_change = next_state - current_state
        cost += np.sum(state_change[:4]**2) * 0.5
        
        return cost
        
    def _compute_cost_to_go(self, current_state: np.ndarray, 
                           target_trajectory: np.ndarray) -> float:
        """计算到目标的启发式成本"""
        if len(target_trajectory) == 0:
            return 0.0
            
        # 使用最近的目标点
        target_state = target_trajectory[0]
        
        # 位置距离
        position_distance = np.linalg.norm(current_state[:2] - target_state[:2])
        
        # 速度差异
        velocity_diff = abs(current_state[2] - target_state[2])
        
        # 总启发式成本
        cost_to_go = position_distance + velocity_diff * 0.1
        
        return cost_to_go
        
    def _select_best_branch(self, node_ids: List[str]) -> str:
        """选择最优分支"""
        if not node_ids:
            return self.root_node.node_id
            
        best_node_id = node_ids[0]
        best_cost = float('inf')
        
        for node_id in node_ids:
            node = self.graph.nodes[node_id]['node']
            if node.total_cost < best_cost:
                best_cost = node.total_cost
                best_node_id = node_id
                
        return best_node_id
        
    def _prune_tree(self) -> Dict[str, Any]:
        """剪枝轨迹树"""
        nodes_pruned = 0
        nodes_to_remove = []
        
        # 找到需要剪枝的节点
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['node']
            
            # 剪枝条件
            if (node.total_cost > self.prune_threshold or 
                node.probability < self.min_probability):
                nodes_to_remove.append(node_id)
                
        # 移除节点
        for node_id in nodes_to_remove:
            # 检查是否有子节点
            if not self.graph.successors(node_id):
                self.graph.remove_node(node_id)
                nodes_pruned += 1
                
        # 更新叶节点列表
        self.leaf_nodes = [n for n in self.graph.nodes 
                          if not self.graph.successors(n)]
        
        return {
            'nodes_pruned': nodes_pruned
        }
        
    def _normalize_angle(self, angle: float) -> float:
        """归一化角度到[-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle
        
    def get_optimal_trajectory(self) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        获取最优轨迹
        
        Returns:
            (states, controls, total_cost): 最优轨迹
        """
        if not self.root_node:
            return [], [], 0.0
            
        # 找到最优叶节点
        best_leaf_id = None
        best_cost = float('inf')
        
        for leaf_id in self.leaf_nodes:
            leaf_node = self.graph.nodes[leaf_id]['node']
            if leaf_node.total_cost < best_cost:
                best_cost = leaf_node.total_cost
                best_leaf_id = leaf_id
                
        if best_leaf_id is None:
            return [], [], 0.0
            
        # 回溯路径
        path = self._find_path(self.root_node.node_id, best_leaf_id)
        
        states = []
        controls = []
        
        for node_id in path:
            node = self.graph.nodes[node_id]['node']
            states.append(node.state)
            controls.append(node.control)
            
        return states, controls, best_cost
        
    def _find_path(self, start_id: str, end_id: str) -> List[str]:
        """查找路径"""
        # 简化版本的路径查找
        path = []
        current_id = end_id
        
        while current_id is not None:
            path.append(current_id)
            node = self.graph.nodes[current_id]['node']
            current_id = node.parent_id
            
        path.reverse()
        return path
        
    def get_tree_statistics(self) -> Dict[str, Any]:
        """获取轨迹树统计信息"""
        return {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_leaf_nodes': len(self.leaf_nodes),
            'num_scenarios': len(self.scenario_branches),
            'max_depth': self._compute_tree_depth(),
            'average_branching_factor': self._compute_average_branching_factor()
        }
        
    def _compute_tree_depth(self) -> int:
        """计算树深度"""
        if not self.root_node:
            return 0
            
        max_depth = 0
        for leaf_id in self.leaf_nodes:
            path = self._find_path(self.root_node.node_id, leaf_id)
            max_depth = max(max_depth, len(path) - 1)
            
        return max_depth
        
    def _compute_average_branching_factor(self) -> float:
        """计算平均分支因子"""
        if self.graph.number_of_nodes() <= 1:
            return 0.0
            
        total_branches = 0
        branch_nodes = 0
        
        for node_id in self.graph.nodes:
            successors = self.graph.successors(node_id)
            if len(successors) > 0:
                total_branches += len(successors)
                branch_nodes += 1
                
        return total_branches / max(1, branch_nodes)
        
    def visualize_tree(self) -> Dict[str, Any]:
        """可视化轨迹树结构"""
        tree_data = {
            'nodes': [],
            'edges': []
        }
        
        for node_id in self.graph.nodes:
            node = self.graph.nodes[node_id]['node']
            tree_data['nodes'].append({
                'id': node_id,
                'state': node.state.tolist(),
                'cost': node.total_cost,
                'probability': node.probability,
                'scenario_id': node.scenario_id
            })
            
        for source in self.graph.edges:
            for target in self.graph.edges[source]:
                tree_data['edges'].append({
                    'source': source,
                    'target': target
                })
            
        return tree_data
