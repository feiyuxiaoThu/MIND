"""
多模态处理模块

处理多模态预测的生成、组合和分析。
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .scenario_tree import AgentPrediction, ScenarioData
from .uncertainty import UncertaintyModel
from ..utils.geometry import GeometryUtils


class MultimodalProcessor:
    """多模态处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.uncertainty_model = UncertaintyModel(config)
        
        # 多模态参数
        self.max_modes = config.get('max_modes', 6)
        self.probability_threshold = config.get('probability_threshold', 0.05)
        self.diversity_threshold = config.get('diversity_threshold', 2.0)
        
    def generate_multimodal_predictions(self, base_prediction: AgentPrediction, 
                                      num_modes: Optional[int] = None) -> List[AgentPrediction]:
        """生成多模态预测"""
        if num_modes is None:
            num_modes = self.max_modes
            
        predictions = []
        
        # 基础模态（原始预测）
        predictions.append(base_prediction)
        
        # 生成变异模态
        for i in range(1, num_modes):
            variant_prediction = self._generate_variant_prediction(base_prediction, i)
            predictions.append(variant_prediction)
            
        # 归一化概率
        total_prob = sum(p.probability for p in predictions)
        for pred in predictions:
            pred.probability /= total_prob
            
        return predictions
        
    def _generate_variant_prediction(self, base_prediction: AgentPrediction, 
                                   variant_idx: int) -> AgentPrediction:
        """生成变异预测"""
        means = base_prediction.means.copy()
        covariances = base_prediction.covariances.copy()
        
        # 添加系统性偏移
        if variant_idx % 3 == 1:
            # 左偏移
            means[:, 0] += np.sin(np.linspace(0, np.pi/4, len(means))) * 2.0
        elif variant_idx % 3 == 2:
            # 右偏移
            means[:, 0] -= np.sin(np.linspace(0, np.pi/4, len(means))) * 2.0
            
        # 添加速度变化
        if variant_idx % 2 == 0:
            # 减速模态
            speed_factor = 0.8
            for t in range(1, len(means)):
                displacement = means[t] - means[t-1]
                means[t] = means[t-1] + displacement * speed_factor
        else:
            # 加速模态
            speed_factor = 1.2
            for t in range(1, len(means)):
                displacement = means[t] - means[t-1]
                means[t] = means[t-1] + displacement * speed_factor
                
        # 增加不确定性
        covariances *= 1.2
        
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=base_prediction.probability / self.max_modes
        )
        
    def combine_multimodal_predictions(self, predictions: List[AgentPrediction]) -> AgentPrediction:
        """组合多模态预测"""
        if not predictions:
            raise ValueError("Cannot combine empty predictions")
            
        # 使用不确定性模型进行组合
        weights = self.uncertainty_model.adaptive_uncertainty_weighting(predictions)
        combined = self.uncertainty_model.combine_uncertainties(predictions, weights)
        
        return combined
        
    def analyze_mode_diversity(self, predictions: List[AgentPrediction]) -> Dict[str, float]:
        """分析模态多样性"""
        if len(predictions) < 2:
            return {'diversity_score': 0.0, 'mean_distance': 0.0}
            
        distances = []
        
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # 计算预测间的平均距离
                pred1, pred2 = predictions[i], predictions[j]
                
                # 终点距离
                end_distance = GeometryUtils.euclidean_distance(
                    pred1.means[-1], pred2.means[-1]
                )
                
                # 整体轨迹距离
                trajectory_distances = [
                    GeometryUtils.euclidean_distance(p1, p2) 
                    for p1, p2 in zip(pred1.means, pred2.means)
                ]
                mean_trajectory_distance = np.mean(trajectory_distances)
                
                # 组合距离指标
                combined_distance = 0.7 * end_distance + 0.3 * mean_trajectory_distance
                distances.append(combined_distance)
                
        diversity_score = np.mean(distances) if distances else 0.0
        
        return {
            'diversity_score': diversity_score,
            'mean_distance': diversity_score,
            'max_distance': np.max(distances) if distances else 0.0,
            'min_distance': np.min(distances) if distances else 0.0
        }
        
    def select_representative_modes(self, predictions: List[AgentPrediction], 
                               num_selected: int) -> List[AgentPrediction]:
        """选择代表性模态"""
        if len(predictions) <= num_selected:
            return predictions
            
        # 按概率排序
        sorted_predictions = sorted(predictions, key=lambda p: p.probability, reverse=True)
        
        # 选择前几个高概率模态
        selected = [sorted_predictions[0]]
        
        # 贪心选择多样性最大的模态
        while len(selected) < num_selected:
            best_candidate = None
            best_diversity = -1
            
            for candidate in sorted_predictions[1:]:
                if candidate in selected:
                    continue
                    
                # 计算与已选模态的最小距离
                min_distance = float('inf')
                for selected_pred in selected:
                    distance = GeometryUtils.euclidean_distance(
                        candidate.means[-1], selected_pred.means[-1]
                    )
                    min_distance = min(min_distance, distance)
                    
                # 考虑概率和多样性
                score = candidate.probability * min_distance
                
                if score > best_diversity:
                    best_diversity = score
                    best_candidate = candidate
                    
            if best_candidate:
                selected.append(best_candidate)
            else:
                break
                
        return selected
        
    def compute_mode_probabilities(self, predictions: List[AgentPrediction], 
                                 context: Dict[str, Any]) -> List[float]:
        """基于上下文计算模态概率"""
        probabilities = []
        
        for pred in predictions:
            # 基础概率
            base_prob = pred.probability
            
            # 基于上下文调整概率
            context_factor = self._compute_context_factor(pred, context)
            
            # 基于不确定性调整概率
            uncertainty_metrics = self.uncertainty_model.compute_uncertainty_metrics(pred)
            uncertainty_factor = 1.0 / (1.0 + uncertainty_metrics['trace_uncertainty'])
            
            # 组合因子
            adjusted_prob = base_prob * context_factor * uncertainty_factor
            probabilities.append(adjusted_prob)
            
        # 归一化
        total_prob = sum(probabilities)
        if total_prob > 0:
            probabilities = [p / total_prob for p in probabilities]
            
        return probabilities
        
    def _compute_context_factor(self, prediction: AgentPrediction, 
                              context: Dict[str, Any]) -> float:
        """计算上下文因子"""
        factor = 1.0
        
        # 基于目标车道调整
        if 'target_lane' in context:
            target_lane = context['target_lane']
            final_pos = prediction.means[-1]
            distance = GeometryUtils.distance_to_polyline(final_pos, target_lane)
            
            # 距离越近，因子越大
            distance_factor = np.exp(-distance / 5.0)
            factor *= distance_factor
            
        # 基于周围交通调整
        if 'traffic_density' in context:
            density = context['traffic_density']
            # 高密度情况下，倾向于保守预测
            if density > 0.7:
                # 检查预测是否保守（速度较低）
                velocities = np.diff(prediction.means, axis=0)
                speeds = np.linalg.norm(velocities, axis=1)
                mean_speed = np.mean(speeds)
                
                if mean_speed < 10.0:  # m/s
                    factor *= 1.2
                    
        return factor
        
    def cluster_similar_modes(self, predictions: List[AgentPrediction], 
                            distance_threshold: float = None) -> List[List[AgentPrediction]]:
        """聚类相似模态"""
        if distance_threshold is None:
            distance_threshold = self.diversity_threshold
            
        clusters = []
        used_indices = set()
        
        for i, pred_i in enumerate(predictions):
            if i in used_indices:
                continue
                
            # 创建新聚类
            cluster = [pred_i]
            used_indices.add(i)
            
            # 查找相似预测
            for j, pred_j in enumerate(predictions):
                if j <= i or j in used_indices:
                    continue
                    
                # 计算相似度
                similarity = self._compute_similarity(pred_i, pred_j)
                
                if similarity > distance_threshold:
                    cluster.append(pred_j)
                    used_indices.add(j)
                    
            clusters.append(cluster)
            
        return clusters
        
    def _compute_similarity(self, pred1: AgentPrediction, 
                          pred2: AgentPrediction) -> float:
        """计算两个预测的相似度"""
        # 终点距离
        end_distance = GeometryUtils.euclidean_distance(pred1.means[-1], pred2.means[-1])
        
        # 轨迹形状相似度
        trajectory_similarity = 0.0
        min_length = min(len(pred1.means), len(pred2.means))
        
        for t in range(min_length):
            distance = GeometryUtils.euclidean_distance(pred1.means[t], pred2.means[t])
            trajectory_similarity += np.exp(-distance / 2.0)
            
        trajectory_similarity /= min_length
        
        # 组合相似度
        overall_similarity = 0.6 * trajectory_similarity + 0.4 * np.exp(-end_distance / 5.0)
        
        return overall_similarity
        
    def generate_scenario_data(self, ego_prediction: AgentPrediction, 
                             exo_predictions: List[AgentPrediction]) -> ScenarioData:
        """生成场景数据"""
        return ScenarioData(
            ego_prediction=ego_prediction,
            exo_predictions=exo_predictions,
            probability=ego_prediction.probability,
            timestamp=0.0,
            metadata={}
        )