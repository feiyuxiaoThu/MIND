"""
场景工厂

根据环境信息自动选择和配置合适的场景处理器。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Type
from .intersection import IntersectionScenario
from .lane_change import LaneChangeScenario
from .highway import HighwayScenario
from ..scenario.scenario_tree import ScenarioData, AgentPrediction


class ScenarioFactory:
    """场景工厂类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 注册场景处理器
        self.scenario_handlers = {
            'intersection': IntersectionScenario,
            'lane_change': LaneChangeScenario,
            'highway': HighwayScenario
        }
        
        # 场景优先级（用于多重场景检测）
        self.scenario_priority = ['intersection', 'lane_change', 'highway']
        
    def detect_scenario(self, ego_state: np.ndarray,
                       road_data: Dict[str, Any],
                       exo_predictions: List[AgentPrediction]) -> str:
        """检测当前场景类型"""
        scenario_scores = {}
        
        # 检测每种场景
        for scenario_type, handler_class in self.scenario_handlers.items():
            handler = handler_class(self.config.get(scenario_type, {}))
            
            if scenario_type == 'intersection':
                detected = handler.detect_intersection(ego_state, road_data)
            elif scenario_type == 'lane_change':
                detected = handler.detect_lane_change_situation(ego_state, road_data)
            elif scenario_type == 'highway':
                detected = handler.detect_highway_scenario(ego_state, road_data)
            else:
                detected = False
                
            scenario_scores[scenario_type] = 1.0 if detected else 0.0
            
        # 选择最高优先级的检测场景
        for scenario_type in self.scenario_priority:
            if scenario_scores[scenario_type] > 0.5:
                return scenario_type
                
        return 'default'
        
    def create_scenario_handler(self, scenario_type: str) -> Any:
        """创建场景处理器"""
        if scenario_type in self.scenario_handlers:
            handler_class = self.scenario_handlers[scenario_type]
            return handler_class(self.config.get(scenario_type, {}))
        else:
            # 返回默认处理器（使用高速场景作为基础）
            return HighwayScenario(self.config.get('highway', {}))
            
    def get_scenario_specific_config(self, scenario_type: str,
                                   base_config: Dict[str, Any],
                                   ego_state: np.ndarray,
                                   road_data: Dict[str, Any],
                                   exo_predictions: List[AgentPrediction]) -> Dict[str, Any]:
        """获取场景特定配置"""
        handler = self.create_scenario_handler(scenario_type)
        adjusted_config = base_config.copy()
        
        # 根据场景类型调整配置
        if scenario_type == 'intersection':
            # 路口场景配置
            intersection_type = handler.classify_intersection_type(road_data)
            
            # 调整AIME参数
            if 'aime' not in adjusted_config:
                adjusted_config['aime'] = {}
            adjusted_config['aime']['max_depth'] = 6  # 路口需要更深的搜索
            adjusted_config['aime']['uncertainty_threshold'] = 8.0
            
            # 调整成本权重
            if 'cost' not in adjusted_config:
                adjusted_config['cost'] = {}
            if 'safety' not in adjusted_config['cost']:
                adjusted_config['cost']['safety'] = {}
            adjusted_config['cost']['safety']['weight'] = 10.0  # 增加安全权重
            
        elif scenario_type == 'lane_change':
            # 换道场景配置
            change_type = handler.classify_lane_change_type(ego_state, road_data)
            
            # 调整规划视野
            adjusted_config['horizon'] = 60  # 换道需要更长的视野
            
            # 调整成本权重
            if 'cost' not in adjusted_config:
                adjusted_config['cost'] = {}
            if 'comfort' not in adjusted_config['cost']:
                adjusted_config['cost']['comfort'] = {}
            adjusted_config['cost']['comfort']['weight'] = 2.0  # 增加舒适性权重
            
        elif scenario_type == 'highway':
            # 高速场景配置
            traffic_density = handler.classify_traffic_density(ego_state, exo_predictions)
            
            # 根据密度调整参数
            if traffic_density == 'heavy':
                adjusted_config['horizon'] = 40  # 密集交通缩短视野
                if 'target_velocity' in adjusted_config:
                    adjusted_config['target_velocity'] *= 0.8
            elif traffic_density == 'light':
                if 'target_velocity' in adjusted_config:
                    adjusted_config['target_velocity'] *= 1.1
                    
        return adjusted_config
        
    def generate_scenario_predictions(self, scenario_type: str,
                                   ego_state: np.ndarray,
                                   road_data: Dict[str, Any],
                                   exo_predictions: List[AgentPrediction]) -> List[AgentPrediction]:
        """生成场景特定的预测"""
        handler = self.create_scenario_handler(scenario_type)
        
        if scenario_type == 'intersection':
            return handler.generate_intersection_predictions(ego_state, road_data)
        elif scenario_type == 'lane_change':
            return handler.generate_lane_change_predictions(ego_state, road_data)
        elif scenario_type == 'highway':
            traffic_density = handler.classify_traffic_density(ego_state, exo_predictions)
            return handler.generate_highway_predictions(ego_state, traffic_density)
        else:
            return exo_predictions  # 默认返回原始预测
            
    def evaluate_scenario_safety(self, scenario_type: str,
                               ego_prediction: AgentPrediction,
                               exo_predictions: List[AgentPrediction],
                               road_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估场景安全性"""
        handler = self.create_scenario_handler(scenario_type)
        
        if scenario_type == 'intersection':
            return handler.evaluate_gap_safety(ego_prediction, exo_predictions)
        elif scenario_type == 'lane_change':
            return handler.evaluate_lane_change_safety(ego_prediction, exo_predictions, road_data)
        elif scenario_type == 'highway':
            return handler.evaluate_highway_safety(ego_prediction, exo_predictions)
        else:
            return {'safe_to_proceed': True, 'recommended_action': 'maintain'}
            
    def adjust_config_for_safety(self, scenario_type: str,
                               base_config: Dict[str, Any],
                               safety_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """根据安全评估调整配置"""
        handler = self.create_scenario_handler(scenario_type)
        return handler.adjust_planning_parameters(base_config, safety_assessment)
        
    def generate_target_trajectory(self, scenario_type: str,
                                ego_state: np.ndarray,
                                target_lane: np.ndarray,
                                road_data: Dict[str, Any]) -> np.ndarray:
        """生成场景特定的目标轨迹"""
        handler = self.create_scenario_handler(scenario_type)
        
        if scenario_type == 'intersection':
            intersection_type = handler.classify_intersection_type(road_data)
            return handler.generate_target_trajectory(ego_state, target_lane, intersection_type)
        elif scenario_type == 'lane_change':
            return handler.generate_lane_change_trajectory(ego_state, road_data)
        elif scenario_type == 'highway':
            # 需要交通密度信息，这里使用默认值
            traffic_density = 'moderate'
            return handler.generate_highway_trajectory(ego_state, target_lane, traffic_density)
        else:
            # 默认轨迹（直线）
            horizon = self.config.get('horizon', 50)
            dt = self.config.get('dt', 0.1)
            target_trajectory = np.zeros((horizon, 6))
            
            x, y, v, theta, a, delta = ego_state
            
            for t in range(horizon):
                target_pos = np.array([
                    x + v * t * dt * np.cos(theta),
                    y + v * t * dt * np.sin(theta)
                ])
                target_trajectory[t] = [target_pos[0], target_pos[1], v, theta, 0.0, 0.0]
                
            return target_trajectory
            
    def register_scenario_handler(self, scenario_type: str, handler_class: Type):
        """注册新的场景处理器"""
        self.scenario_handlers[scenario_type] = handler_class
        
    def get_available_scenarios(self) -> List[str]:
        """获取可用的场景类型"""
        return list(self.scenario_handlers.keys())
        
    def create_composite_handler(self, scenario_types: List[str]) -> 'CompositeScenarioHandler':
        """创建复合场景处理器"""
        return CompositeScenarioHandler(scenario_types, self.scenario_handlers, self.config)


class CompositeScenarioHandler:
    """复合场景处理器，用于处理多重场景"""
    
    def __init__(self, scenario_types: List[str], handlers: Dict[str, Type],
                 config: Dict[str, Any]):
        self.scenario_types = scenario_types
        self.handlers = {}
        
        # 创建处理器实例
        for scenario_type in scenario_types:
            if scenario_type in handlers:
                self.handlers[scenario_type] = handlers[scenario_type](
                    config.get(scenario_type, {})
                )
                
    def evaluate_combined_safety(self, ego_prediction: AgentPrediction,
                               exo_predictions: List[AgentPrediction],
                               road_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估复合场景安全性"""
        combined_assessment = {
            'safe_to_proceed': True,
            'recommended_action': 'maintain',
            'scenario_assessments': {},
            'overall_risk': 'low'
        }
        
        risk_levels = []
        
        for scenario_type, handler in self.handlers.items():
            if scenario_type == 'intersection':
                assessment = handler.evaluate_gap_safety(ego_prediction, exo_predictions)
            elif scenario_type == 'lane_change':
                assessment = handler.evaluate_lane_change_safety(
                    ego_prediction, exo_predictions, road_data
                )
            elif scenario_type == 'highway':
                assessment = handler.evaluate_highway_safety(ego_prediction, exo_predictions)
            else:
                assessment = {'safe_to_proceed': True, 'recommended_action': 'maintain'}
                
            combined_assessment['scenario_assessments'][scenario_type] = assessment
            
            # 收集风险等级
            if not assessment['safe_to_proceed']:
                combined_assessment['safe_to_proceed'] = False
                
                # 推荐最保守的行动
                if assessment['recommended_action'] == 'emergency_brake':
                    combined_assessment['recommended_action'] = 'emergency_brake'
                elif (assessment['recommended_action'] == 'decelerate' and 
                      combined_assessment['recommended_action'] != 'emergency_brake'):
                    combined_assessment['recommended_action'] = 'decelerate'
                    
                # 记录风险等级
                if 'collision_risk' in assessment:
                    risk_levels.append(assessment['collision_risk'])
                    
        # 确定整体风险等级
        if risk_levels:
            if 'high' in risk_levels:
                combined_assessment['overall_risk'] = 'high'
            elif 'medium' in risk_levels:
                combined_assessment['overall_risk'] = 'medium'
            else:
                combined_assessment['overall_risk'] = 'low'
                
        return combined_assessment