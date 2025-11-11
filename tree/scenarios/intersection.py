"""
路口场景适配

专门处理路口场景的规划逻辑。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from abc import ABC, abstractmethod
from ..scenario.scenario_tree import ScenarioData, AgentPrediction
from ..utils.geometry import GeometryUtils


class IntersectionScenario:
    """路口场景处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 路口参数
        self.intersection_size = config.get('intersection_size', 20.0)  # 路口大小
        self.approach_distance = config.get('approach_distance', 50.0)  # 接近距离
        self.stop_line_distance = config.get('stop_line_distance', 5.0)  # 停止线距离
        self.crossing_speed = config.get('crossing_speed', 5.0)  # 通过速度
        
        # 安全参数
        self.min_gap_time = config.get('min_gap_time', 3.0)  # 最小间隙时间
        self.safety_margin = config.get('safety_margin', 2.0)  # 安全边距
        
    def detect_intersection(self, ego_state: np.ndarray, road_data: Dict[str, Any]) -> bool:
        """检测是否在路口场景"""
        ego_position = ego_state[:2]
        
        # 检查是否在路口区域内
        if 'intersection_center' in road_data:
            center = road_data['intersection_center']
            distance = GeometryUtils.euclidean_distance(ego_position, center)
            return distance < self.intersection_size
            
        # 检查是否接近路口
        if 'intersection_boundaries' in road_data:
            for boundary in road_data['intersection_boundaries']:
                distance = GeometryUtils.distance_to_polyline(ego_position, boundary)
                if distance < self.approach_distance:
                    return True
                    
        return False
        
    def classify_intersection_type(self, road_data: Dict[str, Any]) -> str:
        """分类路口类型"""
        if 'intersection_type' in road_data:
            return road_data['intersection_type']
            
        # 基于道路连接数判断
        if 'incoming_lanes' in road_data:
            num_lanes = len(road_data['incoming_lanes'])
            if num_lanes == 4:
                return 'four_way'
            elif num_lanes == 3:
                return 'three_way'
            else:
                return 'complex'
                
        return 'unknown'
        
    def generate_target_trajectory(self, ego_state: np.ndarray, 
                                 target_lane: np.ndarray,
                                 intersection_type: str) -> np.ndarray:
        """生成路口目标轨迹"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        target_trajectory = np.zeros((horizon, 6))
        
        # 当前状态
        x, y, v, theta, a, delta = ego_state
        
        # 根据路口类型生成不同的通过策略
        if intersection_type == 'four_way':
            target_trajectory = self._generate_four_way_trajectory(
                ego_state, target_lane, horizon, dt
            )
        elif intersection_type == 'three_way':
            target_trajectory = self._generate_three_way_trajectory(
                ego_state, target_lane, horizon, dt
            )
        else:
            # 默认轨迹
            target_trajectory = self._generate_default_trajectory(
                ego_state, target_lane, horizon, dt
            )
            
        return target_trajectory
        
    def _generate_four_way_trajectory(self, ego_state: np.ndarray, target_lane: np.ndarray,
                                    horizon: int, dt: float) -> np.ndarray:
        """生成四路口通过轨迹"""
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        # 分阶段生成轨迹
        # 1. 接近阶段（减速）
        approach_time = 2.0  # 秒
        approach_steps = int(approach_time / dt)
        
        # 2. 观察阶段（停止或慢行）
        observation_time = 1.0  # 秒
        observation_steps = int(observation_time / dt)
        
        # 3. 通过阶段（加速）
        crossing_time = 4.0  # 秒
        crossing_steps = int(crossing_time / dt)
        
        for t in range(horizon):
            if t < approach_steps:
                # 减速接近
                progress = t / approach_steps
                target_velocity = v * (1 - 0.5 * progress)
                target_acceleration = -2.0
                
            elif t < approach_steps + observation_steps:
                # 观察阶段
                target_velocity = 2.0  # 慢速
                target_acceleration = 0.0
                
            elif t < approach_steps + observation_steps + crossing_steps:
                # 通过路口
                progress = (t - approach_steps - observation_steps) / crossing_steps
                target_velocity = 2.0 + (self.crossing_speed - 2.0) * progress
                target_acceleration = 1.0
                
            else:
                # 恢复正常速度
                target_velocity = self.crossing_speed
                target_acceleration = 0.0
                
            # 计算目标位置（沿目标车道）
            if t < len(target_lane):
                target_pos = target_lane[t]
                target_heading = theta  # 简化处理
            else:
                target_pos = target_lane[-1]
                target_heading = theta
                
            target_trajectory[t] = [
                target_pos[0], target_pos[1], target_velocity,
                target_heading, target_acceleration, 0.0
            ]
            
        return target_trajectory
        
    def _generate_three_way_trajectory(self, ego_state: np.ndarray, target_lane: np.ndarray,
                                     horizon: int, dt: float) -> np.ndarray:
        """生成三路口通过轨迹"""
        # 类似四路口，但通过时间更短
        return self._generate_four_way_trajectory(ego_state, target_lane, horizon, dt)
        
    def _generate_default_trajectory(self, ego_state: np.ndarray, target_lane: np.ndarray,
                                   horizon: int, dt: float) -> np.ndarray:
        """生成默认轨迹"""
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        for t in range(horizon):
            if t < len(target_lane):
                target_pos = target_lane[t]
            else:
                target_pos = target_lane[-1]
                
            target_velocity = min(v, self.crossing_speed)
            target_heading = theta
            target_acceleration = 0.0
            
            target_trajectory[t] = [
                target_pos[0], target_pos[1], target_velocity,
                target_heading, target_acceleration, 0.0
            ]
            
        return target_trajectory
        
    def evaluate_gap_safety(self, ego_prediction: AgentPrediction,
                           exo_predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """评估路口间隙安全性"""
        safety_assessment = {
            'safe_to_cross': True,
            'gap_time': float('inf'),
            'conflicting_vehicles': [],
            'recommended_action': 'proceed'
        }
        
        ego_trajectory = ego_prediction.means
        
        for exo_pred in exo_predictions:
            exo_trajectory = exo_pred.means
            
            # 检查轨迹冲突
            conflict_time = self._detect_trajectory_conflict(ego_trajectory, exo_trajectory)
            
            if conflict_time is not None:
                safety_assessment['safe_to_cross'] = False
                safety_assessment['conflicting_vehicles'].append({
                    'conflict_time': conflict_time,
                    'vehicle_trajectory': exo_trajectory
                })
                
                # 计算间隙时间
                gap_time = conflict_time * self.config.get('dt', 0.1)
                safety_assessment['gap_time'] = min(safety_assessment['gap_time'], gap_time)
                
        # 推荐行动
        if not safety_assessment['safe_to_cross']:
            if safety_assessment['gap_time'] < self.min_gap_time:
                safety_assessment['recommended_action'] = 'stop'
            else:
                safety_assessment['recommended_action'] = 'yield'
        else:
            safety_assessment['recommended_action'] = 'proceed'
            
        return safety_assessment
        
    def _detect_trajectory_conflict(self, traj1: np.ndarray, traj2: np.ndarray) -> Optional[float]:
        """检测轨迹冲突"""
        min_distance = float('inf')
        conflict_time = None
        
        for t in range(min(len(traj1), len(traj2))):
            distance = GeometryUtils.euclidean_distance(traj1[t], traj2[t])
            
            if distance < min_distance:
                min_distance = distance
                conflict_time = t
                
        # 如果最小距离小于安全边距，认为有冲突
        if min_distance < self.safety_margin:
            return conflict_time
            
        return None
        
    def adjust_planning_parameters(self, base_config: Dict[str, Any],
                                 safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """根据安全评估调整规划参数"""
        adjusted_config = base_config.copy()
        
        if not safety_assessment['safe_to_cross']:
            # 增加安全权重
            if 'cost' in adjusted_config:
                cost_config = adjusted_config['cost']
                if 'safety' in cost_config:
                    cost_config['safety']['weight'] *= 2.0
                    
            # 降低目标速度
            if 'target_velocity' in adjusted_config:
                adjusted_config['target_velocity'] *= 0.5
                
            # 调整AIME参数（增加分支深度）
            if 'aime' in adjusted_config:
                aime_config = adjusted_config['aime']
                aime_config['max_depth'] = min(aime_config['max_depth'] + 1, 8)
                
        return adjusted_config
        
    def generate_intersection_predictions(self, ego_state: np.ndarray,
                                       traffic_state: Dict[str, Any]) -> List[AgentPrediction]:
        """生成路口特定的预测"""
        predictions = []
        
        # 基于交通状态生成不同的交互模态
        if 'traffic_flow' in traffic_state:
            flow_direction = traffic_state['traffic_flow']
            
            # 生成不同通过策略的预测
            strategies = ['aggressive', 'normal', 'conservative']
            
            for strategy in strategies:
                prediction = self._generate_strategy_prediction(
                    ego_state, strategy, flow_direction
                )
                predictions.append(prediction)
                
        return predictions
        
    def _generate_strategy_prediction(self, ego_state: np.ndarray, strategy: str,
                                   flow_direction: str) -> AgentPrediction:
        """生成策略特定的预测"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        means = np.zeros((horizon, 2))
        covariances = np.zeros((horizon, 2, 2))
        
        x, y, v, theta, a, delta = ego_state
        
        # 根据策略调整预测
        if strategy == 'aggressive':
            # 较小的不确定性，较快通过
            speed_factor = 1.2
            uncertainty_factor = 0.8
        elif strategy == 'conservative':
            # 较大的不确定性，较慢通过
            speed_factor = 0.8
            uncertainty_factor = 1.5
        else:  # normal
            speed_factor = 1.0
            uncertainty_factor = 1.0
            
        for t in range(horizon):
            # 生成轨迹
            if strategy == 'aggressive':
                # 直接通过
                means[t] = [x + v * speed_factor * t * dt * np.cos(theta),
                           y + v * speed_factor * t * dt * np.sin(theta)]
            elif strategy == 'conservative':
                # 曲线通过，避开冲突
                offset = 2.0 * np.sin(t * dt)
                means[t] = [x + v * speed_factor * t * dt * np.cos(theta) + offset * np.sin(theta),
                           y + v * speed_factor * t * dt * np.sin(theta) - offset * np.cos(theta)]
            else:
                # 正常通过
                means[t] = [x + v * speed_factor * t * dt * np.cos(theta),
                           y + v * speed_factor * t * dt * np.sin(theta)]
                           
            # 设置协方差
            base_cov = 0.5 * (1 + 0.1 * t)  # 随时间增长
            covariances[t] = base_cov * uncertainty_factor * np.eye(2)
            
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=1.0 / 3.0  # 均匀概率
        )