"""
MIND重构版本 - 不含深度学习依赖的多模态预测和决策框架

该模块提供了MIND算法的核心功能重构，专注于场景树和轨迹树的构建与优化，
支持多模态预测作为输入，不依赖深度学习框架。
"""

from .scenario.scenario_tree import ScenarioTree
from .trajectory.trajectory_tree import TrajectoryTree
from .planners.mind_planner import MINDPlanner

__version__ = "1.0.0"
__all__ = ["ScenarioTree", "TrajectoryTree", "MINDPlanner"]