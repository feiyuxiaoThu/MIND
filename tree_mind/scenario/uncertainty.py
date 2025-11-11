"""
不确定性建模模块

处理多模态预测中的不确定性表示、传播和分析。
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .scenario_tree import AgentPrediction, ScenarioData
from ..utils.geometry import GeometryUtils


class UncertaintyModel:
    """不确定性模型"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 不确定性参数
        self.min_covariance = config.get('min_covariance', 1e-5)
        self.max_covariance = config.get('max_covariance', 10.0)
        self.uncertainty_growth_rate = config.get('uncertainty_growth_rate', 1.1)
        self.time_step = config.get('time_step', 0.1)
        
    def propagate_uncertainty(self, prediction: AgentPrediction, 
                            dynamics_model: Optional[Any] = None) -> AgentPrediction:
        """传播不确定性"""
        means = prediction.means.copy()
        covariances = prediction.covariances.copy()
        
        # 时间传播的不确定性增长
        for t in range(1, len(covariances)):
            # 基础不确定性增长
            covariances[t] = covariances[t] * (self.uncertainty_growth_rate ** (t * self.time_step))
            
            # 确保协方差矩阵的合理性
            covariances[t] = self._regularize_covariance(covariances[t])
            
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=prediction.probability
        )
        
    def combine_uncertainties(self, predictions: List[AgentPrediction], 
                            weights: Optional[np.ndarray] = None) -> AgentPrediction:
        """组合多个预测的不确定性"""
        if not predictions:
            raise ValueError("Cannot combine empty predictions")
            
        if weights is None:
            weights = np.array([p.probability for p in predictions])
        weights = weights / np.sum(weights)
        
        # 加权平均均值
        combined_means = np.average([p.means for p in predictions], 
                                   weights=weights, axis=0)
        
        # 组合协方差（考虑预测间的不确定性）
        combined_covariances = []
        for t in range(combined_means.shape[0]):
            # 内部不确定性（每个预测的协方差加权平均）
            internal_cov = np.average([p.covariances[t] for p in predictions], 
                                     weights=weights, axis=0)
            
            # 外部不确定性（预测间的差异）
            external_cov = np.zeros((2, 2))
            for i, pred_i in enumerate(predictions):
                for j, pred_j in enumerate(predictions):
                    diff = pred_i.means[t] - pred_j.means[t]
                    outer = np.outer(diff, diff)
                    external_cov += weights[i] * weights[j] * outer
                    
            combined_cov = internal_cov + external_cov
            combined_covariances.append(self._regularize_covariance(combined_cov))
            
        combined_probability = np.sum(weights)
        
        return AgentPrediction(
            means=combined_means,
            covariances=np.array(combined_covariances),
            probability=combined_probability
        )
        
    def compute_uncertainty_metrics(self, prediction: AgentPrediction) -> Dict[str, float]:
        """计算不确定性指标"""
        covariances = prediction.covariances
        
        # 迹不确定性（总体不确定性）
        trace_uncertainty = np.mean([np.trace(cov) for cov in covariances])
        
        # 行列式不确定性（体积不确定性）
        det_uncertainty = np.mean([np.linalg.det(cov) for cov in covariances])
        
        # 最大特征值不确定性（主方向不确定性）
        max_eig_uncertainty = np.mean([np.max(np.linalg.eigvals(cov)) for cov in covariances])
        
        # 不确定性增长率
        if len(covariances) > 1:
            initial_trace = np.trace(covariances[0])
            final_trace = np.trace(covariances[-1])
            growth_rate = final_trace / initial_trace if initial_trace > 0 else 1.0
        else:
            growth_rate = 1.0
            
        return {
            'trace_uncertainty': trace_uncertainty,
            'det_uncertainty': det_uncertainty,
            'max_eig_uncertainty': max_eig_uncertainty,
            'growth_rate': growth_rate
        }
        
    def compute_cross_correlation(self, pred1: AgentPrediction, 
                                pred2: AgentPrediction) -> np.ndarray:
        """计算两个预测间的交叉相关性"""
        correlations = []
        
        for t in range(min(len(pred1.means), len(pred2.means))):
            cov1 = pred1.covariances[t]
            cov2 = pred2.covariances[t]
            
            # 计算交叉协方差矩阵
            diff = pred1.means[t] - pred2.means[t]
            cross_cov = np.outer(diff, diff)
            
            # 归一化相关性
            norm_factor = np.sqrt(np.trace(cov1) * np.trace(cov2))
            if norm_factor > 0:
                correlation = cross_cov / norm_factor
            else:
                correlation = np.zeros((2, 2))
                
            correlations.append(correlation)
            
        return np.array(correlations)
        
    def adaptive_uncertainty_weighting(self, predictions: List[AgentPrediction]) -> np.ndarray:
        """自适应不确定性加权"""
        if not predictions:
            return np.array([])
            
        # 计算每个预测的不确定性指标
        uncertainties = []
        for pred in predictions:
            metrics = self.compute_uncertainty_metrics(pred)
            # 使用迹不确定性的倒数作为权重
            weight = 1.0 / (metrics['trace_uncertainty'] + 1e-6)
            uncertainties.append(weight)
            
        # 归一化权重
        uncertainties = np.array(uncertainties)
        uncertainties = uncertainties / np.sum(uncertainties)
        
        return uncertainties
        
    def _regularize_covariance(self, covariance: np.ndarray) -> np.ndarray:
        """正则化协方差矩阵"""
        # 确保对称性
        covariance = (covariance + covariance.T) / 2
        
        # 确保正定性
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.clip(eigenvalues, self.min_covariance, self.max_covariance)
        
        # 重构协方差矩阵
        regularized_cov = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        
        return regularized_cov
        
    def compute_confidence_ellipses(self, prediction: AgentPrediction, 
                                  confidence_level: float = 0.95) -> List[Dict[str, Any]]:
        """计算置信椭圆"""
        from scipy.stats import chi2
        
        # 卡方分布临界值
        chi2_critical = chi2.ppf(confidence_level, df=2)
        
        ellipses = []
        for t, (mean, cov) in enumerate(zip(prediction.means, prediction.covariances)):
            # 特征分解
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            
            # 计算椭圆参数
            angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
            width = 2 * np.sqrt(chi2_critical * eigenvalues[0])
            height = 2 * np.sqrt(chi2_critical * eigenvalues[1])
            
            ellipses.append({
                'center': mean.copy(),
                'width': width,
                'height': height,
                'angle': angle,
                'confidence_level': confidence_level,
                'timestamp': t
            })
            
        return ellipses
        
    def compute_risk_metrics(self, prediction: AgentPrediction, 
                           obstacle_positions: np.ndarray) -> Dict[str, float]:
        """计算风险指标"""
        risks = []
        
        for t, (mean, cov) in enumerate(zip(prediction.means, prediction.covariances)):
            # 计算到每个障碍物的马氏距离
            min_distance = float('inf')
            for obstacle_pos in obstacle_positions:
                distance = GeometryUtils.compute_mahalanobis_distance(
                    obstacle_pos, mean, cov
                )
                min_distance = min(min_distance, distance)
                
            # 转换为风险（距离越小风险越大）
            risk = 1.0 / (min_distance + 1.0)
            risks.append(risk)
            
        return {
            'max_risk': np.max(risks),
            'mean_risk': np.mean(risks),
            'final_risk': risks[-1] if risks else 0.0,
            'risk_trend': np.polyfit(range(len(risks)), risks, 1)[0] if len(risks) > 1 else 0.0
        }
        
    def temporal_uncertainty_analysis(self, prediction: AgentPrediction) -> Dict[str, Any]:
        """时序不确定性分析"""
        covariances = prediction.covariances
        
        # 计算每个时间步的不确定性
        trace_uncertainties = [np.trace(cov) for cov in covariances]
        det_uncertainties = [np.linalg.det(cov) for cov in covariances]
        
        # 拟合不确定性增长模型
        time_steps = np.arange(len(trace_uncertainties))
        
        # 线性拟合
        linear_fit = np.polyfit(time_steps, trace_uncertainties, 1)
        
        # 指数拟合
        if len(trace_uncertainties) > 2 and np.all(trace_uncertainties) > 0:
            log_uncertainties = np.log(trace_uncertainties)
            exp_fit = np.polyfit(time_steps, log_uncertainties, 1)
            exp_growth_rate = np.exp(exp_fit[0])
        else:
            exp_growth_rate = 1.0
            
        return {
            'trace_uncertainties': trace_uncertainties,
            'det_uncertainties': det_uncertainties,
            'linear_growth_rate': linear_fit[0],
            'exp_growth_rate': exp_growth_rate,
            'uncertainty_acceleration': linear_fit[0] if len(linear_fit) > 0 else 0.0
        }