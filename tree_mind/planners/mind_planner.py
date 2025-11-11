"""
MIND规划器主实现

重构版本，去除深度学习依赖，支持多模态预测作为输入。
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from ..scenario.scenario_tree import ScenarioTree, ScenarioData, AgentPrediction
from ..scenario.aime import AIME
from ..scenario.multimodal import MultimodalProcessor
from ..scenario.uncertainty import UncertaintyModel
from ..trajectory.trajectory_tree import TrajectoryTree, TrajectoryState, ControlInput
from ..trajectory.optimizers import ILQROptimizer, MPCOptimizer, CBFOptimizer
from ..trajectory.dynamics import BicycleDynamics
from ..trajectory.costs import CompositeCost, SafetyCost, TargetCost, ComfortCost
from ..utils.geometry import GeometryUtils


class MINDPlanner:
    """MIND规划器重构版本"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # 规划参数
        self.dt = config.get('dt', 0.1)
        self.horizon = config.get('horizon', 50)
        self.optimizer_type = config.get('optimizer_type', 'ilqr')
        
        # 初始化组件
        self._init_components()
        
    def _init_components(self):
        """初始化组件"""
        # 场景树组件
        aime_config = self.config.get('aime', {})
        self.aime = AIME(aime_config)
        
        # 多模态处理器
        multimodal_config = self.config.get('multimodal', {})
        self.multimodal_processor = MultimodalProcessor(multimodal_config)
        
        # 不确定性模型
        uncertainty_config = self.config.get('uncertainty', {})
        self.uncertainty_model = UncertaintyModel(uncertainty_config)
        
        # 动力学模型
        dynamics_config = self.config.get('dynamics', {})
        self.dynamics = BicycleDynamics(dynamics_config)
        
        # 成本函数
        self._init_cost_function()
        
        # 优化器
        self._init_optimizer()
        
    def _init_cost_function(self):
        """初始化成本函数"""
        cost_config = self.config.get('cost', {})
        
        self.cost_function = CompositeCost(cost_config)
        
        # 添加各种成本
        if cost_config.get('use_safety_cost', True):
            safety_cost = SafetyCost(cost_config.get('safety', {}))
            self.cost_function.add_cost_function(safety_cost)
            
        if cost_config.get('use_target_cost', True):
            target_cost = TargetCost(cost_config.get('target', {}))
            self.cost_function.add_cost_function(target_cost)
            
        if cost_config.get('use_comfort_cost', True):
            comfort_cost = ComfortCost(cost_config.get('comfort', {}))
            self.cost_function.add_cost_function(comfort_cost)
            
    def _init_optimizer(self):
        """初始化优化器"""
        optimizer_config = self.config.get('optimizer', {})
        
        if self.optimizer_type == 'ilqr':
            self.optimizer = ILQROptimizer(self.dynamics, self.cost_function, optimizer_config)
        elif self.optimizer_type == 'mpc':
            self.optimizer = MPCOptimizer(self.dynamics, self.cost_function, optimizer_config)
        elif self.optimizer_type == 'cbf':
            self.optimizer = CBFOptimizer(self.dynamics, self.cost_function, optimizer_config)
        else:
            raise ValueError(f"Unknown optimizer type: {self.optimizer_type}")
            
    def plan(self, current_state: np.ndarray, multimodal_predictions: List[AgentPrediction],
            target_lane: np.ndarray, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        主规划函数
        
        Args:
            current_state: 当前状态 [x, y, v, theta, a, delta]
            multimodal_predictions: 多模态预测列表
            target_lane: 目标车道
            context: 上下文信息
            
        Returns:
            规划结果字典
        """
        # 1. 生成场景数据
        scenario_data_list = self._generate_scenario_data(multimodal_predictions, context)
        
        # 2. 构建场景树
        scenario_tree = self.aime.generate_scenario_tree(scenario_data_list, target_lane)
        
        # 3. 生成目标轨迹
        target_trajectory = self._generate_target_trajectory(current_state, target_lane)
        
        # 4. 轨迹优化
        trajectory_tree = self.optimizer.optimize_trajectory_tree(
            scenario_tree, current_state, target_trajectory
        )
        
        # 5. 选择最优轨迹
        best_trajectory = trajectory_tree.get_best_trajectory()
        probabilistic_best = trajectory_tree.get_probabilistic_best_trajectory()
        
        # 6. 提取结果
        result = {
            'scenario_tree': scenario_tree,
            'trajectory_tree': trajectory_tree,
            'best_trajectory': best_trajectory,
            'probabilistic_best': probabilistic_best,
            'statistics': {
                'scenario_stats': scenario_tree.get_statistics(),
                'trajectory_stats': trajectory_tree.get_statistics()
            }
        }
        
        return result
        
    def _generate_scenario_data(self, multimodal_predictions: List[AgentPrediction],
                              context: Optional[Dict[str, Any]] = None) -> List[ScenarioData]:
        """生成场景数据"""
        scenario_data_list = []
        
        # 如果只有一个预测，生成多模态变体
        if len(multimodal_predictions) == 1:
            base_prediction = multimodal_predictions[0]
            multimodal_predictions = self.multimodal_processor.generate_multimodal_predictions(
                base_prediction
            )
            
        # 为每个预测创建场景数据
        for ego_prediction in multimodal_predictions:
            # 生成外部智能体预测（简化处理）
            exo_predictions = self._generate_exo_predictions(ego_prediction, context)
            
            # 创建场景数据
            scenario_data = self.multimodal_processor.generate_scenario_data(
                ego_prediction, exo_predictions
            )
            
            scenario_data_list.append(scenario_data)
            
        return scenario_data_list
        
    def _generate_exo_predictions(self, ego_prediction: AgentPrediction,
                                context: Optional[Dict[str, Any]] = None) -> List[AgentPrediction]:
        """生成外部智能体预测"""
        exo_predictions = []
        
        # 简化处理：基于自车预测生成外部智能体预测
        num_agents = context.get('num_agents', 3) if context else 3
        
        for i in range(num_agents):
            # 创建外部智能体预测（基于自车预测的变体）
            exo_means = ego_prediction.means.copy()
            exo_covariances = ego_prediction.covariances.copy()
            
            # 添加偏移和扰动
            offset_angle = 2 * np.pi * i / num_agents
            offset_distance = 5.0
            
            for t in range(len(exo_means)):
                # 添加位置偏移
                exo_means[t, 0] += offset_distance * np.cos(offset_angle)
                exo_means[t, 1] += offset_distance * np.sin(offset_angle)
                
                # 增加不确定性
                exo_covariances[t] *= 1.5
                
            exo_prediction = AgentPrediction(
                means=exo_means,
                covariances=exo_covariances,
                probability=1.0 / num_agents
            )
            
            exo_predictions.append(exo_prediction)
            
        return exo_predictions
        
    def _generate_target_trajectory(self, current_state: np.ndarray, 
                                 target_lane: np.ndarray) -> np.ndarray:
        """生成目标轨迹"""
        horizon = self.horizon
        target_trajectory = np.zeros((horizon, 6))  # [x, y, v, theta, a, delta]
        
        # 当前状态
        x, y, v, theta, a, delta = current_state
        
        # 找到最近的目标点
        distances = [GeometryUtils.euclidean_distance([x, y], point) for point in target_lane]
        closest_idx = np.argmin(distances)
        
        # 沿目标车道生成轨迹
        for t in range(horizon):
            # 目标位置
            target_idx = min(closest_idx + t * 2, len(target_lane) - 1)
            target_pos = target_lane[target_idx]
            
            # 目标速度（可配置）
            target_velocity = self.config.get('target_velocity', 10.0)
            
            # 目标航向（沿车道方向）
            if target_idx < len(target_lane) - 1:
                next_pos = target_lane[target_idx + 1]
                target_heading = np.arctan2(next_pos[1] - target_pos[1], 
                                           next_pos[0] - target_pos[0])
            else:
                target_heading = theta
                
            # 目标加速度和转向角（设为0）
            target_acceleration = 0.0
            target_steering = 0.0
            
            target_trajectory[t] = [
                target_pos[0], target_pos[1], target_velocity,
                target_heading, target_acceleration, target_steering
            ]
            
        return target_trajectory
        
    def get_control_sequence(self, trajectory: List) -> np.ndarray:
        """从轨迹中提取控制序列"""
        if not trajectory:
            return np.array([])
            
        controls = []
        for node in trajectory[1:]:  # 跳过根节点
            control = node.trajectory_data.control
            controls.append([control.acceleration, control.steering_rate])
            
        return np.array(controls)
        
    def get_next_control(self, result: Dict[str, Any]) -> np.ndarray:
        """获取下一个控制输入"""
        best_trajectory = result.get('best_trajectory')
        if not best_trajectory or len(best_trajectory) < 2:
            return np.array([0.0, 0.0])  # 默认控制
            
        # 返回第一个控制输入
        control_node = best_trajectory[1]
        control = control_node.trajectory_data.control
        
        return np.array([control.acceleration, control.steering_rate])
        
    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        self.config.update(new_config)
        
        # 重新初始化组件
        self._init_components()
        
    def switch_optimizer(self, optimizer_type: str):
        """切换优化器"""
        if optimizer_type != self.optimizer_type:
            self.optimizer_type = optimizer_type
            self._init_optimizer()
            print(f"Switched to {optimizer_type} optimizer")
            
    def validate_plan(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """验证规划结果"""
        validation_result = {
            'valid': True,
            'issues': [],
            'warnings': []
        }
        
        best_trajectory = result.get('best_trajectory')
        if not best_trajectory:
            validation_result['valid'] = False
            validation_result['issues'].append("No trajectory found")
            return validation_result
            
        # 验证轨迹
        trajectory_arrays = result['trajectory_tree'].extract_trajectory_arrays(best_trajectory)
        
        # 检查速度合理性
        velocities = trajectory_arrays['velocities']
        if np.any(velocities < 0) or np.any(velocities > 30):
            validation_result['warnings'].append("Unrealistic velocities detected")
            
        # 检查加速度合理性
        if 'controls' in trajectory_arrays:
            accelerations = trajectory_arrays['controls'][:, 0]
            if np.any(np.abs(accelerations) > 5):
                validation_result['warnings'].append("High acceleration detected")
                
        # 检查轨迹连续性
        positions = trajectory_arrays['positions']
        for i in range(1, len(positions)):
            distance = np.linalg.norm(positions[i] - positions[i-1])
            max_reasonable = velocities[i-1] * self.dt * 2
            if distance > max_reasonable:
                validation_result['issues'].append(f"Trajectory discontinuity at step {i}")
                validation_result['valid'] = False
                
        return validation_result