"""
动态分支点分析器

实现MARC论文中的动态分支点算法。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional


class BranchPointAnalyzer:
    """动态分支点分析器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 分支点参数
        self.divergence_threshold = config.get('divergence_threshold', 0.5)
        self.max_branch_time = config.get('max_branch_time', 20)
        self.min_branch_interval = config.get('min_branch_interval', 2)
        self.state_weights = config.get('state_weights', np.array([1.0, 1.0, 0.1, 0.1, 0.01, 0.01]))
        
    def find_optimal_branch_time(self, scenario_trajectories: List[np.ndarray]) -> int:
        """
        找到最优分支时间
        
        Args:
            scenario_trajectories: 场景轨迹列表
            
        Returns:
            optimal_branch_time: 最优分支时间步
        """
        if len(scenario_trajectories) < 2:
            return 0
            
        max_divergence_time = 0
        min_divergence_value = self.divergence_threshold
        
        # 计算所有轨迹对之间的发散
        for i in range(len(scenario_trajectories)):
            for j in range(i + 1, len(scenario_trajectories)):
                divergence = self._compute_trajectory_divergence(
                    scenario_trajectories[i], scenario_trajectories[j]
                )
                
                # 找到发散超过阈值的时间
                for t, div in enumerate(divergence):
                    if div > min_divergence_value and t > max_divergence_time:
                        max_divergence_time = t
                        
        return min(max_divergence_time, self.max_branch_time)
        
    def _compute_trajectory_divergence(self, traj1: np.ndarray, traj2: np.ndarray) -> np.ndarray:
        """计算两条轨迹的发散度"""
        min_length = min(len(traj1), len(traj2))
        divergence = np.zeros(min_length)
        
        for t in range(min_length):
            # 计算加权状态差异
            state_diff = traj1[t] - traj2[t]
            weighted_diff = state_diff * self.state_weights
            
            # 使用欧几里得距离作为发散度量
            divergence[t] = np.linalg.norm(weighted_diff)
            
        return divergence
        
    def analyze_divergence(self, scenario_tree) -> List[int]:
        """
        分析场景分歧
        
        Args:
            scenario_tree: 场景树
            
        Returns:
            divergence_points: 分歧点列表
        """
        # 简化实现：返回默认分歧点
        return [10, 20, 30, 40]
        
    def select_optimal_branch_points(self, divergence_points: List[int],
                                   target_trajectory: np.ndarray) -> List[int]:
        """
        选择最优分支点
        
        Args:
            divergence_points: 分歧点列表
            target_trajectory: 目标轨迹
            
        Returns:
            optimal_points: 最优分支点列表
        """
        # 简化实现：返回前3个分歧点
        return divergence_points[:3]
        
    def analyze_branching_suitability(self, scenarios: List[np.ndarray], 
                                   branch_time: int) -> Dict[str, Any]:
        """
        分析分支适用性
        
        Args:
            scenarios: 场景列表
            branch_time: 分支时间
            
        Returns:
            analysis_result: 分析结果
        """
        if branch_time <= 0 or branch_time >= len(scenarios[0]):
            return {
                'suitable': False,
                'reason': 'Invalid branch time',
                'divergence_at_branch': 0.0
            }
            
        # 计算分支时间的发散度
        divergences_at_branch = []
        for i in range(len(scenarios)):
            for j in range(i + 1, len(scenarios)):
                if branch_time < len(scenarios[i]) and branch_time < len(scenarios[j]):
                    div = self._compute_trajectory_divergence(
                        scenarios[i], scenarios[j]
                    )
                    divergences_at_branch.append(div)
                    
        avg_divergence = np.mean(divergences_at_branch) if divergences_at_branch else 0.0
        
        # 评估分支适用性
        suitable = avg_divergence >= self.divergence_threshold
        
        return {
            'suitable': suitable,
            'reason': 'Low divergence' if not suitable else 'Adequate divergence',
            'divergence_at_branch': avg_divergence,
            'num_scenarios': len(scenarios),
            'branch_time': branch_time
        }
        
    def compute_branching_confidence(self, scenarios: List[np.ndarray], 
                                     branch_time: int) -> float:
        """
        计算分支置信度
        
        Args:
            scenarios: 场景列表
            branch_time: 分支时间
            
        Returns:
            confidence: 分支置信度 [0, 1]
        """
        if not scenarios or branch_time <= 0:
            return 0.0
            
        # 计算分支时间的一致性
        branch_states = []
        for scenario in scenarios:
            if branch_time < len(scenario):
                branch_states.append(scenario[branch_time])
                
        if len(branch_states) < 2:
            return 0.0
            
        # 计算状态方差
        states_array = np.array(branch_states)
        state_variances = np.var(states_array, axis=0)
        
        # 加权方差（位置和速度更重要）
        weighted_variance = np.sum(state_variances * self.state_weights)
        
        # 转换为置信度（方差越小，置信度越高）
        confidence = 1.0 / (1.0 + weighted_variance)
        
        return confidence
        
    def recommend_branching_strategy(self, scenarios: List[np.ndarray]) -> Dict[str, Any]:
        """
        推荐分支策略
        
        Args:
            scenarios: 场景列表
            
        Returns:
            recommendation: 分支建议
        """
        if len(scenarios) < 2:
            return {
                'recommendation': 'no_branch',
                'reason': 'Insufficient scenarios',
                'optimal_time': 0
            }
            
        # 找到最优分支时间
        optimal_time = self.find_optimal_branch_time(scenarios)
        
        # 分析分支适用性
        analysis = self.analyze_branching_suitability(scenarios, optimal_time)
        
        # 计算置信度
        confidence = self.compute_branching_confidence(scenarios, optimal_time)
        
        # 生成建议
        if analysis['suitable'] and confidence > 0.5:
            recommendation = {
                'recommendation': 'branch',
                'reason': 'Suitable divergence and confidence',
                'optimal_time': optimal_time,
                'confidence': confidence,
                'divergence_at_branch': analysis['divergence_at_branch']
            }
        else:
            recommendation = {
                'recommendation': 'delay_branch',
                'reason': 'Low divergence or confidence',
                'optimal_time': optimal_time,
                'confidence': confidence,
                'divergence_at_branch': analysis['divergence_at_branch']
            }
            
        return recommendation
        
    def adjust_branch_parameters(self, divergence_history: List[float]) -> Dict[str, Any]:
        """
        根据历史发散调整分支参数
        
        Args:
            divergence_history: 历史发散值列表
            
        Returns:
            adjusted_params: 调整后的参数
        """
        if not divergence_history:
            return {}
            
        # 计算平均发散和趋势
        avg_divergence = np.mean(divergence_history)
        recent_divergence = np.mean(divergence_history[-5:]) if len(divergence_history) >= 5 else avg_divergence
        
        # 调整发散阈值
        if recent_divergence > avg_divergence * 1.5:
            # 发散增加，提高阈值
            new_threshold = self.divergence_threshold * 1.1
        elif recent_divergence < avg_divergence * 0.5:
            # 发散减少，降低阈值
            new_threshold = self.divergence_threshold * 0.9
        else:
            new_threshold = self.divergence_threshold
            
        # 调整最大分支时间
        if avg_divergence > 2.0:
            # 高发散，减少分支时间
            new_max_time = max(self.max_branch_time - 2, 5)
        elif avg_divergence < 0.5:
            # 低发散，增加分支时间
            new_max_time = min(self.max_branch_time + 2, 30)
        else:
            new_max_time = self.max_branch_time
            
        return {
            'divergence_threshold': new_threshold,
            'max_branch_time': new_max_time,
            'avg_divergence': avg_divergence,
            'recent_divergence': recent_divergence
        }