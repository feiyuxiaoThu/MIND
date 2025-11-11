"""
换道场景适配

专门处理换道场景的规划逻辑。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..scenario.scenario_tree import ScenarioData, AgentPrediction
from ..utils.geometry import GeometryUtils


class LaneChangeScenario:
    """换道场景处理器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 换道参数
        self.lane_width = config.get('lane_width', 3.5)  # 车道宽度
        self.min_change_distance = config.get('min_change_distance', 30.0)  # 最小换道距离
        self.max_change_distance = config.get('max_change_distance', 100.0)  # 最大换道距离
        self.change_duration = config.get('change_duration', 5.0)  # 换道持续时间
        self.lateral_acceleration = config.get('lateral_acceleration', 2.0)  # 横向加速度
        
        # 安全参数
        self.min_gap_distance = config.get('min_gap_distance', 10.0)  # 最小间隙距离
        self.min_gap_time = config.get('min_gap_time', 2.0)  # 最小间隙时间
        self.safety_margin = config.get('safety_margin', 1.5)  # 安全边距
        
    def detect_lane_change_situation(self, ego_state: np.ndarray,
                                   road_data: Dict[str, Any]) -> bool:
        """检测换道场景"""
        # 检查是否有目标车道
        if 'target_lane' not in road_data:
            return False
            
        # 检查当前车道和目标车道
        current_lane = road_data.get('current_lane')
        target_lane = road_data.get('target_lane')
        
        if current_lane is None or target_lane is None:
            return False
            
        # 检查是否需要换道
        if current_lane == target_lane:
            return False
            
        return True
        
    def classify_lane_change_type(self, ego_state: np.ndarray,
                                road_data: Dict[str, Any]) -> str:
        """分类换道类型"""
        current_lane = road_data.get('current_lane')
        target_lane = road_data.get('target_lane')
        
        if current_lane is None or target_lane is None:
            return 'unknown'
            
        # 基于车道编号判断换道方向
        try:
            current_id = int(current_lane.split('_')[-1])
            target_id = int(target_lane.split('_')[-1])
            
            if target_id > current_id:
                return 'left_change'
            elif target_id < current_id:
                return 'right_change'
            else:
                return 'same_lane'
        except:
            return 'unknown'
            
    def generate_lane_change_trajectory(self, ego_state: np.ndarray,
                                      road_data: Dict[str, Any]) -> np.ndarray:
        """生成换道轨迹"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        # 换道类型
        change_type = self.classify_lane_change_type(ego_state, road_data)
        
        # 获取车道信息
        current_lane = road_data.get('current_lane_center', [])
        target_lane = road_data.get('target_lane_center', [])
        
        if not current_lane or not target_lane:
            return self._generate_default_trajectory(ego_state, horizon, dt)
            
        # 生成换道轨迹
        if change_type == 'left_change':
            target_trajectory = self._generate_left_change_trajectory(
                ego_state, current_lane, target_lane, horizon, dt
            )
        elif change_type == 'right_change':
            target_trajectory = self._generate_right_change_trajectory(
                ego_state, current_lane, target_lane, horizon, dt
            )
        else:
            target_trajectory = self._generate_default_trajectory(ego_state, horizon, dt)
            
        return target_trajectory
        
    def _generate_left_change_trajectory(self, ego_state: np.ndarray,
                                      current_lane: np.ndarray, target_lane: np.ndarray,
                                      horizon: int, dt: float) -> np.ndarray:
        """生成左换道轨迹"""
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        # 换道阶段
        preparation_steps = int(1.0 / dt)  # 准备阶段
        change_steps = int(self.change_duration / dt)  # 换道阶段
        stabilization_steps = int(1.0 / dt)  # 稳定阶段
        
        for t in range(horizon):
            if t < preparation_steps:
                # 准备阶段：调整位置和速度
                progress = t / preparation_steps
                lateral_offset = 0.2 * progress  # 开始横向偏移
                target_velocity = v * (1 + 0.1 * progress)  # 略微加速
                
            elif t < preparation_steps + change_steps:
                # 换道阶段：横向移动
                progress = (t - preparation_steps) / change_steps
                
                # 使用正弦函数生成平滑的横向轨迹
                lateral_offset = self.lane_width * (0.5 - 0.5 * np.cos(np.pi * progress))
                target_velocity = v * 1.1  # 保持略高速度
                
            elif t < preparation_steps + change_steps + stabilization_steps:
                # 稳定阶段：调整到目标车道中心
                progress = (t - preparation_steps - change_steps) / stabilization_steps
                lateral_offset = self.lane_width * (1 - 0.5 * progress)
                target_velocity = v * (1.1 - 0.1 * progress)  # 恢复正常速度
                
            else:
                # 正常行驶
                lateral_offset = self.lane_width
                target_velocity = v
                
            # 计算目标位置
            if t < len(current_lane):
                base_position = current_lane[t]
            else:
                # 外推当前位置
                last_pos = current_lane[-1]
                last_vel = (current_lane[-1] - current_lane[-2]) / dt
                base_position = last_pos + last_vel * (t - len(current_lane) + 1) * dt
                
            # 添加横向偏移
            target_position = base_position + np.array([0, lateral_offset])
            
            # 计算目标航向
            if t > 0:
                prev_position = target_trajectory[t-1, :2]
                target_heading = np.arctan2(
                    target_position[1] - prev_position[1],
                    target_position[0] - prev_position[0]
                )
            else:
                target_heading = theta
                
            target_trajectory[t] = [
                target_position[0], target_position[1], target_velocity,
                target_heading, 0.0, 0.0
            ]
            
        return target_trajectory
        
    def _generate_right_change_trajectory(self, ego_state: np.ndarray,
                                       current_lane: np.ndarray, target_lane: np.ndarray,
                                       horizon: int, dt: float) -> np.ndarray:
        """生成右换道轨迹"""
        # 类似左换道，但横向偏移为负
        target_trajectory = self._generate_left_change_trajectory(
            ego_state, current_lane, target_lane, horizon, dt
        )
        
        # 反转横向偏移
        for t in range(horizon):
            lateral_offset = target_trajectory[t, 1] - current_lane[min(t, len(current_lane)-1), 1]
            target_trajectory[t, 1] -= 2 * lateral_offset  # 反转偏移
            
        return target_trajectory
        
    def _generate_default_trajectory(self, ego_state: np.ndarray,
                                   horizon: int, dt: float) -> np.ndarray:
        """生成默认轨迹"""
        target_trajectory = np.zeros((horizon, 6))
        
        x, y, v, theta, a, delta = ego_state
        
        for t in range(horizon):
            # 保持当前速度和方向
            target_position = np.array([
                x + v * t * dt * np.cos(theta),
                y + v * t * dt * np.sin(theta)
            ])
            
            target_trajectory[t] = [
                target_position[0], target_position[1], v,
                theta, 0.0, 0.0
            ]
            
        return target_trajectory
        
    def evaluate_lane_change_safety(self, ego_prediction: AgentPrediction,
                                  exo_predictions: List[AgentPrediction],
                                  road_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估换道安全性"""
        safety_assessment = {
            'safe_to_change': True,
            'gap_available': True,
            'conflicting_vehicles': [],
            'recommended_action': 'proceed',
            'optimal_change_time': 0.0
        }
        
        ego_trajectory = ego_prediction.means
        
        # 检查目标车道上的车辆
        target_lane_vehicles = []
        for exo_pred in exo_predictions:
            if self._is_vehicle_in_target_lane(exo_pred, road_data):
                target_lane_vehicles.append(exo_pred)
                
        # 评估间隙
        if target_lane_vehicles:
            gap_analysis = self._analyze_gaps(ego_trajectory, target_lane_vehicles)
            safety_assessment.update(gap_analysis)
            
            # 推荐行动
            if not safety_assessment['gap_available']:
                safety_assessment['recommended_action'] = 'wait'
            elif safety_assessment['min_gap_time'] < self.min_gap_time:
                safety_assessment['recommended_action'] = 'yield'
            else:
                safety_assessment['recommended_action'] = 'proceed'
                
        return safety_assessment
        
    def _is_vehicle_in_target_lane(self, vehicle_prediction: AgentPrediction,
                                 road_data: Dict[str, Any]) -> bool:
        """判断车辆是否在目标车道"""
        target_lane = road_data.get('target_lane_center', [])
        if not target_lane:
            return False
            
        # 检查车辆轨迹是否接近目标车道
        vehicle_trajectory = vehicle_prediction.means
        
        for position in vehicle_trajectory:
            distance = GeometryUtils.distance_to_polyline(position, target_lane)
            if distance < self.lane_width / 2:
                return True
                
        return False
        
    def _analyze_gaps(self, ego_trajectory: np.ndarray,
                     target_lane_vehicles: List[AgentPrediction]) -> Dict[str, Any]:
        """分析车道间隙"""
        gap_analysis = {
            'gap_available': True,
            'min_gap_distance': float('inf'),
            'min_gap_time': float('inf'),
            'optimal_change_time': 0.0,
            'conflicting_vehicles': []
        }
        
        for vehicle in target_lane_vehicles:
            vehicle_trajectory = vehicle.means
            
            # 计算最小距离和时间间隙
            min_distance = float('inf')
            min_time_gap = float('inf')
            
            for t, (ego_pos, vehicle_pos) in enumerate(zip(ego_trajectory, vehicle_trajectory)):
                distance = GeometryUtils.euclidean_distance(ego_pos, vehicle_pos)
                min_distance = min(min_distance, distance)
                
                # 计算时间间隙（基于相对速度）
                if t > 0:
                    ego_vel = np.linalg.norm(ego_trajectory[t] - ego_trajectory[t-1])
                    vehicle_vel = np.linalg.norm(vehicle_trajectory[t] - vehicle_trajectory[t-1])
                    relative_vel = abs(ego_vel - vehicle_vel)
                    
                    if relative_vel > 0.1:
                        time_gap = distance / relative_vel
                        min_time_gap = min(min_time_gap, time_gap)
                        
            # 检查是否安全
            if min_distance < self.min_gap_distance:
                gap_analysis['gap_available'] = False
                gap_analysis['conflicting_vehicles'].append({
                    'min_distance': min_distance,
                    'min_time_gap': min_time_gap
                })
                
            gap_analysis['min_gap_distance'] = min(gap_analysis['min_gap_distance'], min_distance)
            gap_analysis['min_gap_time'] = min(gap_analysis['min_gap_time'], min_time_gap)
            
        return gap_analysis
        
    def adjust_planning_parameters(self, base_config: Dict[str, Any],
                                 safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """根据安全评估调整规划参数"""
        adjusted_config = base_config.copy()
        
        if not safety_assessment['gap_available']:
            # 增加安全权重
            if 'cost' in adjusted_config:
                cost_config = adjusted_config['cost']
                if 'safety' in cost_config:
                    cost_config['safety']['weight'] *= 3.0
                    
            # 降低目标速度
            if 'target_velocity' in adjusted_config:
                adjusted_config['target_velocity'] *= 0.7
                
            # 延长规划视野
            if 'horizon' in adjusted_config:
                adjusted_config['horizon'] = min(adjusted_config['horizon'] + 20, 100)
                
        return adjusted_config
        
    def generate_lane_change_predictions(self, ego_state: np.ndarray,
                                      road_data: Dict[str, Any]) -> List[AgentPrediction]:
        """生成换道特定的预测"""
        predictions = []
        
        change_type = self.classify_lane_change_type(ego_state, road_data)
        
        # 生成不同激进程度的换道预测
        strategies = ['aggressive', 'normal', 'conservative']
        
        for strategy in strategies:
            prediction = self._generate_change_strategy_prediction(
                ego_state, change_type, strategy, road_data
            )
            predictions.append(prediction)
            
        return predictions
        
    def _generate_change_strategy_prediction(self, ego_state: np.ndarray, change_type: str,
                                          strategy: str, road_data: Dict[str, Any]) -> AgentPrediction:
        """生成换道策略特定的预测"""
        horizon = self.config.get('horizon', 50)
        dt = self.config.get('dt', 0.1)
        
        means = np.zeros((horizon, 2))
        covariances = np.zeros((horizon, 2, 2))
        
        x, y, v, theta, a, delta = ego_state
        
        # 根据策略和换道类型调整参数
        if strategy == 'aggressive':
            speed_factor = 1.2
            duration_factor = 0.8  # 更快的换道
            uncertainty_factor = 0.7
        elif strategy == 'conservative':
            speed_factor = 0.9
            duration_factor = 1.3  # 更慢的换道
            uncertainty_factor = 1.5
        else:  # normal
            speed_factor = 1.0
            duration_factor = 1.0
            uncertainty_factor = 1.0
            
        # 换道方向
        lateral_direction = 1 if change_type == 'left_change' else -1
        
        # 生成轨迹
        change_duration = self.change_duration * duration_factor
        change_steps = int(change_duration / dt)
        
        for t in range(horizon):
            if t < change_steps:
                # 换道阶段
                progress = t / change_steps
                
                # 横向位置（使用平滑的S曲线）
                lateral_offset = lateral_direction * self.lane_width * (
                    0.5 - 0.5 * np.cos(np.pi * progress)
                )
                
                # 纵向位置
                longitudinal_distance = v * speed_factor * t * dt
            else:
                # 换道完成
                lateral_offset = lateral_direction * self.lane_width
                longitudinal_distance = v * speed_factor * t * dt
                
            # 计算位置
            means[t] = [
                x + longitudinal_distance * np.cos(theta) - lateral_offset * np.sin(theta),
                y + longitudinal_distance * np.sin(theta) + lateral_offset * np.cos(theta)
            ]
            
            # 设置协方差
            base_cov = 0.3 * (1 + 0.05 * t)
            covariances[t] = base_cov * uncertainty_factor * np.eye(2)
            
        return AgentPrediction(
            means=means,
            covariances=covariances,
            probability=1.0 / 3.0
        )