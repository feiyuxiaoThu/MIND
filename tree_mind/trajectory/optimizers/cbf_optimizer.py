"""
CBF优化器实现

控制障碍函数优化器，提供安全保证。
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from ..dynamics import DynamicsModel
from ..costs import CostFunction
from ..trajectory_tree import TrajectoryTree, TrajectoryNode, TrajectoryData, TrajectoryState, ControlInput


class ControlBarrierFunction:
    """控制障碍函数"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.alpha = config.get('alpha', 1.0)  # CBF参数
        self.safety_distance = config.get('safety_distance', 2.0)
        
    def evaluate(self, state: np.ndarray, obstacle_position: np.ndarray) -> float:
        """评估CBF值"""
        position = state[:2]
        distance = np.linalg.norm(position - obstacle_position)
        return distance - self.safety_distance
        
    def gradient(self, state: np.ndarray, obstacle_position: np.ndarray) -> np.ndarray:
        """计算CBF梯度"""
        position = state[:2]
        distance = np.linalg.norm(position - obstacle_position)
        
        if distance < 1e-6:
            return np.zeros_like(state)
            
        grad = np.zeros_like(state)
        grad[:2] = (position - obstacle_position) / distance
        return grad
        
    def lie_derivative(self, state: np.ndarray, control: np.ndarray, 
                      obstacle_position: np.ndarray, dynamics: DynamicsModel) -> float:
        """计算李导数"""
        h = self.evaluate(state, obstacle_position)
        grad_h = self.gradient(state, obstacle_position)
        
        # 状态导数
        A, B = dynamics.get_jacobian(state, control)
        state_dot = A @ state + B @ control
        
        return grad_h @ state_dot
        
    def constraint(self, state: np.ndarray, control: np.ndarray,
                  obstacle_position: np.ndarray, dynamics: DynamicsModel) -> float:
        """CBF约束"""
        h = self.evaluate(state, obstacle_position)
        lie_h = self.lie_derivative(state, control, obstacle_position, dynamics)
        
        return lie_h + self.alpha * h


class CBFOptimizer:
    """CBF优化器"""
    
    def __init__(self, dynamics_model: DynamicsModel, cost_function: CostFunction,
                 config: Dict[str, Any]):
        self.dynamics = dynamics_model
        self.cost = cost_function
        self.config = config
        
        # CBF参数
        self.cbf_list = []
        self.obstacles = []
        
        # 优化参数
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        self.learning_rate = config.get('learning_rate', 0.1)
        
        # QP求解参数
        self.use_qp = config.get('use_qp', True)  # 默认使用QP求解
        
        # 状态和控制维度
        self.state_dim = 6
        self.control_dim = 2
        
    def add_barrier_function(self, cbf: ControlBarrierFunction):
        """添加障碍函数"""
        self.cbf_list.append(cbf)
        
    def set_obstacles(self, obstacles: List[np.ndarray]):
        """设置障碍物"""
        self.obstacles = obstacles
        
    def optimize(self, initial_state: np.ndarray, initial_controls: np.ndarray,
                 target_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        CBF优化
        
        Args:
            initial_state: 初始状态
            initial_controls: 初始控制序列
            target_trajectory: 目标轨迹
            
        Returns:
            optimized_states: 优化后的状态序列
            optimized_controls: 优化后的控制序列
        """
        import time
        
        # 如果启用QP且CVXPY可用，使用QP求解
        if self.use_qp:
            try:
                start_time = time.time()
                states, controls = self.optimize_with_qp(initial_state, initial_controls, target_trajectory)
                qp_time = time.time() - start_time
                print(f"CBF QP optimization completed in {qp_time*1000:.2f} ms")
                return states, controls
            except ImportError:
                print("CVXPY not available, falling back to gradient descent")
        
        # 使用梯度下降方法
        start_time = time.time()
        controls = initial_controls.copy()
        states = self._simulate_trajectory(initial_state, controls)
        
        # 迭代优化
        for iteration in range(self.max_iterations):
            # 计算梯度
            grad = self._compute_gradient(states, controls, target_trajectory)
            
            # 投影梯度下降（考虑CBF约束）
            projected_grad = self._project_gradient(states, controls, grad)
            
            # 更新控制
            new_controls = controls - self.learning_rate * projected_grad
            
            # 重新模拟
            new_states = self._simulate_trajectory(initial_state, new_controls)
            
            # 检查收敛
            if np.linalg.norm(new_controls - controls) < self.tolerance:
                print(f"CBF optimizer converged at iteration {iteration}")
                break
                
            controls = new_controls
            states = new_states
        
        gradient_time = time.time() - start_time
        print(f"CBF gradient descent optimization completed in {gradient_time*1000:.2f} ms")
        
        return states, controls
        
    def _simulate_trajectory(self, initial_state: np.ndarray, controls: np.ndarray) -> np.ndarray:
        """模拟轨迹"""
        states = np.zeros((len(controls) + 1, self.state_dim))
        states[0] = initial_state
        
        for t, control in enumerate(controls):
            states[t + 1] = self.dynamics.step(states[t], control)
            
        return states
        
    def _compute_gradient(self, states: np.ndarray, controls: np.ndarray,
                         target_trajectory: Optional[np.ndarray] = None) -> np.ndarray:
        """计算成本梯度"""
        gradient = np.zeros_like(controls)
        
        for t in range(len(controls)):
            target = target_trajectory[t] if target_trajectory is not None else None
            
            # 使用数值方法计算梯度
            eps = 1e-6
            base_cost = self.cost.compute(states[t], controls[t], target)
            
            for i in range(self.control_dim):
                control_plus = controls[t].copy()
                control_plus[i] += eps
                
                cost_plus = self.cost.compute(states[t], control_plus, target)
                gradient[t, i] = (cost_plus - base_cost) / eps
                
        return gradient
        
    def _project_gradient(self, states: np.ndarray, controls: np.ndarray,
                         gradient: np.ndarray) -> np.ndarray:
        """投影梯度到可行集"""
        projected_gradient = gradient.copy()
        
        for t in range(len(controls)):
            for cbf in self.cbf_list:
                for obstacle in self.obstacles:
                    # 检查CBF约束
                    constraint_value = cbf.constraint(states[t], controls[t], obstacle, self.dynamics)
                    
                    if constraint_value < 0:  # 违反约束
                        # 计算约束梯度
                        constraint_grad = self._compute_constraint_gradient(
                            states[t], controls[t], obstacle, cbf
                        )
                        
                        # 投影梯度
                        if np.linalg.norm(constraint_grad) > 1e-6:
                            constraint_grad = constraint_grad / np.linalg.norm(constraint_grad)
                            projection = np.dot(gradient[t], constraint_grad)
                            
                            if projection > 0:  # 梯度指向不可行区域
                                projected_gradient[t] -= projection * constraint_grad
                                
        return projected_gradient
        
    def _compute_constraint_gradient(self, state: np.ndarray, control: np.ndarray,
                                   obstacle: np.ndarray, cbf: ControlBarrierFunction) -> np.ndarray:
        """计算约束梯度"""
        eps = 1e-6
        base_constraint = cbf.constraint(state, control, obstacle, self.dynamics)
        constraint_grad = np.zeros(self.control_dim)
        
        for i in range(self.control_dim):
            control_plus = control.copy()
            control_plus[i] += eps
            
            constraint_plus = cbf.constraint(state, control_plus, obstacle, self.dynamics)
            constraint_grad[i] = (constraint_plus - base_constraint) / eps
            
        return constraint_grad
        
    def optimize_with_qp(self, initial_state: np.ndarray, initial_controls: np.ndarray,
                        target_trajectory: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """使用二次规划求解CBF-QP问题"""
        import time
        start_time = time.time()
        
        try:
            import cvxpy as cp
        except ImportError:
            print("CVXPY not available, falling back to gradient descent")
            self.use_qp = False
            return self.optimize(initial_state, initial_controls, target_trajectory)
            
        horizon = len(initial_controls)
        
        # 定义决策变量
        u = cp.Variable((horizon, self.control_dim))
        
        # 目标函数（二次成本）
        cost = 0
        for t in range(horizon):
            # 简化的二次成本 - 最小化控制输入
            cost += cp.quad_form(u[t], np.eye(self.control_dim))
                
        # CBF约束
        constraints = []
        states = [initial_state]
        
        for t in range(horizon):
            # 预测下一状态 - 使用线性化近似
            if t == 0:
                current_state = initial_state
            else:
                # 使用前一步的状态和控制来近似
                A, B = self.dynamics.get_jacobian(current_state, initial_controls[t-1])
                current_state = A @ current_state + B @ initial_controls[t-1]
            
            # 简化的安全约束 - 保持与障碍物的最小距离
            for obstacle in self.obstacles:
                # 计算到障碍物的距离
                distance = cp.norm(current_state[:2] - obstacle)
                # 安全距离约束
                constraints.append(distance >= self.config.get('safety_distance', 2.0))
                    
            # 控制输入约束
            constraints.append(cp.norm(u[t], 'inf') <= 5.0)  # 限制控制输入大小
                    
        # 求解
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(verbose=False, solver=cp.ECOS)
        
        if problem.status == 'optimal':
            optimized_controls = u.value
            optimized_states = self._simulate_trajectory(initial_state, optimized_controls)
            return optimized_states, optimized_controls
        else:
            print(f"QP solve failed: {problem.status}")
            return self.optimize(initial_state, initial_controls, target_trajectory)
            
    def optimize_trajectory_tree(self, scenario_tree, initial_state: np.ndarray,
                                target_trajectory: np.ndarray) -> TrajectoryTree:
        """优化轨迹树"""
        trajectory_tree = TrajectoryTree(self.config)
        
        # 添加根节点
        trajectory_tree.add_root(
            TrajectoryState(
                position=initial_state[:2],
                velocity=initial_state[2],
                heading=initial_state[3],
                acceleration=initial_state[4],
                steering_angle=initial_state[5],
                timestamp=0.0
            ),
            ControlInput(acceleration=0.0, steering_rate=0.0)
        )
        
        # 获取场景分支
        scenario_branches = scenario_tree.get_scenario_branches()
        
        for branch in scenario_branches:
            # 提取场景数据
            scenario_data = branch[-1].scenario_data
            
            # 设置障碍物（从外部智能体预测）
            obstacles = []
            for exo_pred in scenario_data.exo_predictions:
                obstacles.append(exo_pred.means[-1])
                
            # 创建CBF
            self.obstacles = obstacles
            self.cbf_list = []
            for _ in obstacles:
                cbf = ControlBarrierFunction(self.config)
                self.cbf_list.append(cbf)
                
            # 初始化控制
            horizon = len(scenario_data.ego_prediction.means)
            initial_controls = np.zeros((horizon, self.control_dim))
            
            # CBF优化 - 测试三种方法并比较耗时
            import time
            
            # 方法1: 梯度下降
            self.use_qp = False
            start_time = time.time()
            states_gd, controls_gd = self.optimize(initial_state, initial_controls, target_trajectory)
            gd_time = time.time() - start_time
            
            # 方法2: QP求解
            self.use_qp = True
            try:
                start_time = time.time()
                states_qp, controls_qp = self.optimize_with_qp(initial_state, initial_controls, target_trajectory)
                qp_time = time.time() - start_time
                use_qp_result = True
            except Exception as e:
                print(f"QP method failed: {e}")
                qp_time = float('inf')
                use_qp_result = False
                states_qp, controls_qp = states_gd, controls_gd
            
            # 方法3: 混合方法（先QP失败则用GD）
            self.use_qp = True
            start_time = time.time()
            try:
                states_hybrid, controls_hybrid = self.optimize(initial_state, initial_controls, target_trajectory)
            except:
                self.use_qp = False
                states_hybrid, controls_hybrid = self.optimize(initial_state, initial_controls, target_trajectory)
            hybrid_time = time.time() - start_time
            
            # 打印耗时比较
            print(f"\nCBF Optimization Method Comparison:")
            print(f"  Gradient Descent: {gd_time*1000:.2f} ms")
            if use_qp_result:
                print(f"  QP Solver: {qp_time*1000:.2f} ms")
            else:
                print(f"  QP Solver: Failed")
            print(f"  Hybrid Method: {hybrid_time*1000:.2f} ms")
            
            # 选择最快的结果
            if use_qp_result and qp_time < gd_time and qp_time < hybrid_time:
                optimized_states, optimized_controls = states_qp, controls_qp
                print(f"  -> Using QP result (fastest)")
            elif hybrid_time < gd_time:
                optimized_states, optimized_controls = states_hybrid, controls_hybrid
                print(f"  -> Using Hybrid result (fastest)")
            else:
                optimized_states, optimized_controls = states_gd, controls_gd
                print(f"  -> Using Gradient Descent result (fastest)")
            
            # 构建轨迹树
            parent_id = "root"
            for t in range(1, min(len(optimized_states), horizon)):
                state = optimized_states[t]
                control = optimized_controls[t-1]
                
                # 计算成本
                target = target_trajectory[t] if target_trajectory is not None else None
                cost = self.cost.compute(state, control, target)
                
                # 添加轨迹节点
                trajectory_node = trajectory_tree.add_trajectory_step(
                    parent_id,
                    TrajectoryState(
                        position=state[:2],
                        velocity=state[2],
                        heading=state[3],
                        acceleration=state[4],
                        steering_angle=state[5],
                        timestamp=t * self.dynamics.dt
                    ),
                    ControlInput(acceleration=control[0], steering_rate=control[1]),
                    cost,
                    scenario_data.probability
                )
                
                parent_id = trajectory_node.key
                
        return trajectory_tree