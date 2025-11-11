"""
高速场景适配

专门处理高速公路场景的规划逻辑。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..scenario.scenario_tree import ScenarioData, AgentPrediction
from ..utils.geometry import GeometryUtils


class HighwayScenario:
    """高速场景处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 高速参数
        self.highway_speed = config.get('highway_speed', 25.0)  # 高速公路速度 (m/s)
        self.min_following_distance = config.get('min_following_distance', 2.0)  # 最小跟车距离 (秒)
        self.lane_width = config.get('lane_width', 3.5)  # 车道宽度
        self.max_acceleration = config.get('max_acceleration', 2.0)  # 最大加速度
        self.max_deceleration = config.get('max_deceleration', 4.0)  # 最大减速度
        
        # 安全参数
        self.safety_time_gap = config.get('safety_time_gap', 2.0)  # 安全时间间隙
        self.emergency_brake_distance = config.get('emergency_brake_distance', 5.0)  # 紧急制动距离
        self.cooperation_factor = config.get('cooperation_factor', 0.8)  # 协作因子
        
    def detect_highway_scenario(self, ego_state: np.ndarray,
                              road_data: Dict[str, Any]) -> bool:
        """检测高速公路场景"""
        # 基于速度判断
        ego_velocity = ego_state[2]
        if ego_velocity > self.highway_speed * 0.7:  # 70%的高速速度
            return True
            
        # 基于道路类型判断
        if 'road_type' in road_data:
            return road_data['road_type'] == 'highway'
            
        # 基于车道数判断（多车道）
        if 'num_lanes' in road_data:
            return road_data['num_lanes'] >= 3
            
        return False
        
    def classify_traffic_density(self, ego_state: np.ndarray,
                               exo_predictions: List[AgentPrediction]) -> str:
        """分类交通密度"""
        ego_position = ego_state[:2]
        nearby_vehicles = 0
        
        # 统计附近车辆
        for exo_pred in exo_predictions:
            vehicle_trajectory = exo_pred.means
            for position in vehicle_trajectory:
                distance = GeometryUtils.euclidean_distance(ego_position, position)
                if distance < 100.0:  # 100米范围
                    nearby_vehicles += 1
                    break
                    
        # 分类密度
        if nearby_vehicles <= 2:
            return 'light'
        elif nearby_vehicles <= 5:
            return 'moderate'
        else:
            return 'heavy'
            
    def generate_highway_trajectory(self, ego_state: np.ndarray,
                                   target_lane: np.ndarray,
                                   traffic_density: str) -> np.ndarray:
        """生成高速场景轨迹"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        # 根据交通密度调整速度
        if traffic_density == 'light':
            target_speed = self.highway_speed
        elif traffic_density == 'moderate':
            target_speed = self.highway_speed * 0.9
        else:  # heavy
            target_speed = self.highway_speed * 0.8
            
        for t in range(horizon):
            # 计算目标位置
            if t < len(target_lane):
                target_pos = target_lane[t]
            else:
                # 沿当前方向继续
                distance = target_speed * t * dt
                target_pos = np.array([
                    x + distance * np.cos(theta),
                    y + distance * np.sin(theta)
                ])
                
            # 平滑速度调整
            if t < 20:  # 前2秒平滑调整
                progress = t / 20
                target_velocity = v + (target_speed - v) * progress
            else:
                target_velocity = target_speed
                
            # 保持当前航向
            target_heading = theta
            
            target_trajectory[t] = [
                target_pos[0], target_pos[1], target_velocity,
                target_heading, 0.0, 0.0
            ]
            
        return target_trajectory
        
    def evaluate_highway_safety(self, ego_prediction: AgentPrediction,
                              exo_predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """评估高速场景安全性"""
        safety_assessment = {
            'safe_to_proceed': True,
            'collision_risk': 'low',
            'recommended_action': 'maintain_speed',
            'critical_vehicles': [],
            'safe_following_distance': True
        }
        
        ego_trajectory = ego_prediction.means
        ego_velocities = self._compute_velocities(ego_trajectory)
        
        # 检查每辆外部车辆
        for exo_pred in exo_predictions:
            exo_trajectory = exo_pred.means
            exo_velocities = self._compute_velocities(exo_trajectory)
            
            # 分析跟车情况
            following_analysis = self._analyze_following_situation(
                ego_trajectory, ego_velocities, exo_trajectory, exo_velocities
            )
            
            if not following_analysis['safe']:
                safety_assessment['safe_to_proceed'] = False
                safety_assessment['critical_vehicles'].append(following_analysis)
                
                if following_analysis['risk_level'] == 'high':
                    safety_assessment['collision_risk'] = 'high'
                    safety_assessment['recommended_action'] = 'emergency_brake'
                elif following_analysis['risk_level'] == 'medium':
                    safety_assessment['collision_risk'] = 'medium'
                    safety_assessment['recommended_action'] = 'decelerate'
                    
        return safety_assessment
        
    def _compute_velocities(self, trajectory: np.ndarray) -> np.ndarray:
        """计算速度序列"""
        velocities = np.zeros(len(trajectory))
        
        for t in range(1, len(trajectory)):
            displacement = trajectory[t] - trajectory[t-1]
            distance = np.linalg.norm(displacement)
            velocities[t] = distance / self.config.get('dt', 0.1)
            
        return velocities
        
    def _analyze_following_situation(self, ego_trajectory: np.ndarray, ego_velocities: np.ndarray,
                                   exo_trajectory: np.ndarray, exo_velocities: np.ndarray) -> Dict[str, Any]:
        """分析跟车情况"""
        analysis = {
            'safe': True,
            'risk_level': 'low',
            'min_distance': float('inf'),
            'min_time_gap': float('inf'),
            'relative_velocity': 0.0
        }
        
        # 检查是否在同一车道且在前方
        for t in range(min(len(ego_trajectory), len(exo_trajectory))):
            ego_pos = ego_trajectory[t]
            exo_pos = exo_trajectory[t]
            
            # 简化的同车道判断（基于横向距离）
            lateral_distance = abs(ego_pos[1] - exo_pos[1])
            if lateral_distance > self.lane_width / 2:
                continue
                
            # 检查是否在前方
            longitudinal_distance = exo_pos[0] - ego_pos[0]
            if longitudinal_distance <= 0:
                continue
                
            # 更新最小距离
            analysis['min_distance'] = min(analysis['min_distance'], longitudinal_distance)
            
            # 计算时间间隙
            ego_vel = ego_velocities[t]
            time_gap = longitudinal_distance / max(ego_vel, 0.1)
            analysis['min_time_gap'] = min(analysis['min_time_gap'], time_gap)
            
            # 计算相对速度
            relative_vel = ego_vel - exo_velocities[t]
            analysis['relative_velocity'] = max(analysis['relative_velocity'], abs(relative_vel))
            
        # 评估安全性
        if analysis['min_distance'] < self.emergency_brake_distance:
            analysis['safe'] = False
            analysis['risk_level'] = 'high'
        elif analysis['min_time_gap'] < self.safety_time_gap:
            analysis['safe'] = False
            analysis['risk_level'] = 'medium'
            
        return analysis
        
    def adjust_planning_parameters(self, base_config: Dict[str, Any],
                                 safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """根据安全评估调整规划参数"""
        adjusted_config = base_config.copy()
        
        if not safety_assessment['safe_to_proceed']:
            # 根据风险等级调整
            if safety_assessment['collision_risk'] == 'high':
                # 紧急制动配置
                if 'dynamics' in adjusted_config:
                    dynamics_config = adjusted_config['dynamics']
                    dynamics_config['max_deceleration'] = self.max_deceleration * 1.5
                    
                # 增加安全权重
                if 'cost' in adjusted_config:
                    cost_config = adjusted_config['cost']
                    if 'safety' in cost_config:
                        cost_config['safety']['weight'] *= 5.0
                        
            elif safety_assessment['collision_risk'] == 'medium':
                # 减速配置
                if 'target_velocity' in adjusted_config:
                    adjusted_config['target_velocity'] *= 0.7
                    
                # 增加安全权重
                if 'cost' in adjusted_config:
                    cost_config = adjusted_config['cost']
                    if 'safety' in cost_config:
                        cost_config['safety']['weight'] *= 2.0
                        
        return adjusted_config
        
    def generate_cooperative_predictions(self, ego_state: np.ndarray,
                                       exo_predictions: List[AgentPrediction]) -> List[AgentPrediction]:
        """生成协作预测"""
        cooperative_predictions = []
        
        # 为每辆外部车辆生成协作预测
        for exo_pred in exo_predictions:
            # 基于自车行为调整外部车辆预测
            cooperative_pred = self._generate_cooperative_vehicle_prediction(
                ego_state, exo_pred
            )
            cooperative_predictions.append(cooperative_pred)
            
        return cooperative_predictions
        
    def _generate_cooperative_vehicle_prediction(self, ego_state: np.ndarray,
                                              exo_prediction: AgentPrediction) -> AgentPrediction:
        """生成协作车辆预测"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        means = exo_prediction.means.copy()
        covariances = exo_prediction.covariances.copy()
        
        ego_velocity = ego_state[2]
        exo_velocities = self._compute_velocities(means)
        
        # 协作调整：如果自车较快，前方车辆可能加速让行
        for t in range(1, len(means)):
            # 检查是否在前方
            longitudinal_distance = means[t, 0] - ego_state[0]
            lateral_distance = abs(means[t, 1] - ego_state[1])
            
            if (longitudinal_distance > 0 and longitudinal_distance < 50 and 
                lateral_distance < self.lane_width):
                
                # 计算速度差
                speed_diff = ego_velocity - exo_velocities[t]
                
                # 如果自车明显更快，前方车辆可能加速
                if speed_diff > 5.0:  # 5m/s速度差
                    cooperation_factor = self.cooperation_factor
                    
                    # 调整位置（加速）
                    acceleration = cooperation_factor * 2.0  # 2m/s²协作加速度
                    means[t, 0] += 0.5 * acceleration * (t * dt) ** 2
                    
                    # 减少不确定性（协作行为更可预测）
                    covariances[t] *= 0.8
                    
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=exo_prediction.probability
        )
        
    def generate_highway_predictions(self, ego_state: np.ndarray,
                                   traffic_density: str) -> List[AgentPrediction]:
        """生成高速场景特定的预测"""
        predictions = []
        
        # 根据交通密度生成不同的行为模式
        if traffic_density == 'light':
            behaviors = ['normal', 'aggressive']
        elif traffic_density == 'moderate':
            behaviors = ['normal', 'conservative', 'cooperative']
        else:  # heavy
            behaviors = ['conservative', 'cooperative']
            
        for behavior in behaviors:
            prediction = self._generate_behavior_prediction(ego_state, behavior)
            predictions.append(prediction)
            
        return predictions
        
    def _generate_behavior_prediction(self, ego_state: np.ndarray,
                                   behavior: str) -> AgentPrediction:
        """生成行为特定的预测"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        means = np.zeros((horizon, 2))
        covariances = np.zeros((horizon, 2, 2))
        
        x, y, v, theta, a, delta = ego_state
        
        # 根据行为类型调整参数
        if behavior == 'aggressive':
            speed_factor = 1.1
            uncertainty_factor = 0.8
            lane_change_prob = 0.3
        elif behavior == 'conservative':
            speed_factor = 0.9
            uncertainty_factor = 1.3
            lane_change_prob = 0.1
        elif behavior == 'cooperative':
            speed_factor = 1.0
            uncertainty_factor = 0.9
            lane_change_prob = 0.2
        else:  # normal
            speed_factor = 1.0
            uncertainty_factor = 1.0
            lane_change_prob = 0.15
            
        for t in range(horizon):
            # 基础位置（保持车道）
            base_distance = v * speed_factor * t * dt
            means[t] = [
                x + base_distance * np.cos(theta),
                y + base_distance * np.sin(theta)
            ]
            
            # 随机换道行为
            if np.random.random() < lane_change_prob * dt:
                lane_change_offset = np.random.choice([-1, 1]) * self.lane_width
                means[t, 1] += lane_change_offset
                
            # 设置协方差
            base_cov = 0.2 * (1 + 0.02 * t)
            covariances[t] = base_cov * uncertainty_factor * np.eye(2)
            
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=1.0 / len(['normal', 'aggressive', 'conservative', 'cooperative'])
        )