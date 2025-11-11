# CBF和MPC算法设计文档

## 概述

本文档详细介绍了MIND项目中使用的控制障碍函数（CBF）和模型预测控制（MPC）算法的具体设计和实现。

---

## 1. 控制障碍函数（CBF）算法

### 1.1 理论基础

控制障碍函数（Control Barrier Function, CBF）是一种用于保证系统安全性的数学工具。其核心思想是通过构造障碍函数来确保系统状态始终保持在安全集合内。

#### 数学定义

给定系统动力学：
```
ẋ = f(x) + g(x)u
```

其中：
- `x ∈ ℝⁿ` 是系统状态
- `u ∈ ℝᵐ` 是控制输入
- `f(x)` 和 `g(x)` 是系统动力学函数

安全集合 C 定义为：
```
C = {x ∈ ℝⁿ | h(x) ≥ 0}
```

其中 `h(x)` 是障碍函数。

#### CBF条件

函数 `h(x)` 是控制障碍函数，如果存在连续函数 `α: ℝ → ℝ` 且 `α(0) = 0`，对于所有 `x ∈ C`：
```
sup_u [L_fh(x) + L_gh(x)u + α(h(x))] ≥ 0
```

其中：
- `L_fh(x) = ∇h(x)ᵀf(x)` 是李导数
- `L_gh(x) = ∇h(x)ᵀg(x)` 是李导数

### 1.2 算法实现

#### 1.2.1 ControlBarrierFunction类

```python
class ControlBarrierFunction:
    def __init__(self, config):
        self.alpha = config.get('alpha', 1.0)           # CBF参数
        self.safety_distance = config.get('safety_distance', 2.0)
```

**核心方法：**

1. **障碍函数评估**
```python
def evaluate(self, state, obstacle_position):
    position = state[:2]
    distance = np.linalg.norm(position - obstacle_position)
    return distance - self.safety_distance
```

2. **梯度计算**
```python
def gradient(self, state, obstacle_position):
    position = state[:2]
    distance = np.linalg.norm(position - obstacle_position)
    
    if distance < 1e-6:
        return np.zeros_like(state)
        
    grad = np.zeros_like(state)
    grad[:2] = (position - obstacle_position) / distance
    return grad
```

3. **CBF约束计算**
```python
def constraint(self, state, control, obstacle_position, dynamics):
    h = self.evaluate(state, obstacle_position)
    lie_h = self.lie_derivative(state, control, obstacle_position, dynamics)
    
    return lie_h + self.alpha * h
```

#### 1.2.2 CBFOptimizer类

**优化策略：**

1. **梯度下降法**
```python
def optimize(self, initial_state, initial_controls, target_trajectory):
    controls = initial_controls.copy()
    states = self._simulate_trajectory(initial_state, controls)
    
    for iteration in range(self.max_iterations):
        # 计算梯度
        grad = self._compute_gradient(states, controls, target_trajectory)
        
        # 投影梯度下降（考虑CBF约束）
        projected_grad = self._project_gradient(states, controls, grad)
        
        # 更新控制
        new_controls = controls - self.learning_rate * projected_grad
        
        # 检查收敛
        if np.linalg.norm(new_controls - controls) < self.tolerance:
            break
            
        controls = new_controls
        states = new_states
```

2. **投影梯度法**
```python
def _project_gradient(self, states, controls, gradient):
    projected_gradient = gradient.copy()
    
    for t in range(len(controls)):
        for cbf in self.cbf_list:
            for obstacle in self.obstacles:
                # 检查CBF约束
                constraint_value = cbf.constraint(states[t], controls[t], 
                                               obstacle, self.dynamics)
                
                if constraint_value < 0:  # 违反约束
                    # 计算约束梯度并投影
                    constraint_grad = self._compute_constraint_gradient(...)
                    
                    if np.linalg.norm(constraint_grad) > 1e-6:
                        constraint_grad = constraint_grad / np.linalg.norm(constraint_grad)
                        projection = np.dot(gradient[t], constraint_grad)
                        
                        if projection > 0:  # 梯度指向不可行区域
                            projected_gradient[t] -= projection * constraint_grad
```

3. **二次规划求解（可选）**
```python
def optimize_with_qp(self, initial_state, initial_controls, target_trajectory):
    try:
        from cvxpy import Variable, Problem, Minimize, quad_form
        
        # 定义决策变量
        u = Variable((horizon, self.control_dim))
        
        # 目标函数
        cost = 0
        for t in range(horizon):
            cost += quad_form(u[t], np.eye(self.control_dim))
            
        # CBF约束
        constraints = []
        for t in range(horizon):
            next_state = self.dynamics.step(states[-1], u[t])
            
            for cbf in self.cbf_list:
                for obstacle in self.obstacles:
                    constraint_value = cbf.evaluate(next_state, obstacle)
                    constraints.append(constraint_value >= 0)
                    
        # 求解QP问题
        problem = Problem(Minimize(cost), constraints)
        problem.solve(verbose=False)
```

### 1.3 算法特点

**优势：**
- **安全保证**：理论上的安全性证明
- **实时性**：计算效率高
- **灵活性**：可处理多种约束

**局限性：**
- **保守性**：可能导致过于保守的行为
- **局部最优**：可能陷入局部最优解
- **参数敏感性**：对α参数敏感

---

## 2. 模型预测控制（MPC）算法

### 2.1 理论基础

模型预测控制是一种基于优化的控制方法，通过在有限时间视野内求解最优控制序列来实现系统控制。

#### 基本原理

在每个时间步 `k`：
1. 测量当前状态 `x(k)`
2. 求解有限视野优化问题：
   ```
   min Σ_{i=0}^{N-1} L(x(k+i|k), u(k+i|k)) + V_f(x(k+N|k))
   s.t. x(k+i+1|k) = f(x(k+i|k), u(k+i|k))
        x(k|k) = x(k)
        (x(k+i|k), u(k+i|k)) ∈ X × U
   ```
3. 应用第一个控制输入 `u*(k) = u*(k|k)`
4. 重复上述过程

### 2.2 算法实现

#### 2.2.1 MPCOptimizer类

**核心参数：**
```python
class MPCOptimizer:
    def __init__(self, dynamics_model, cost_function, config):
        # MPC参数
        self.horizon = config.get('horizon', 20)           # 预测视野
        self.dt = config.get('dt', 0.1)                    # 时间步长
        self.max_iterations = config.get('max_iterations', 100)
        self.tolerance = config.get('tolerance', 1e-6)
        
        # 约束参数
        self.max_acceleration = config.get('max_acceleration', 3.0)
        self.min_acceleration = config.get('min_acceleration', -5.0)
        self.max_steering_rate = config.get('max_steering_rate', 0.5)
        self.min_steering_rate = config.get('min_steering_rate', -0.5)
        self.max_velocity = config.get('max_velocity', 20.0)
        self.min_velocity = config.get('min_velocity', 0.0)
```

#### 2.2.2 优化问题求解

**目标函数：**
```python
def _objective_function(self, control_flat, initial_state, target_trajectory, obstacles):
    controls = control_flat.reshape(-1, self.control_dim)
    states = self._simulate_trajectory(initial_state, controls)
    
    total_cost = 0.0
    
    for t in range(len(controls)):
        target = target_trajectory[t] if target_trajectory is not None else None
        
        # 基础成本
        stage_cost = self.cost.compute(states[t], controls[t], target)
        total_cost += stage_cost
        
        # 障碍物成本
        if obstacles:
            position = states[t][:2]
            for obstacle in obstacles:
                distance = np.linalg.norm(position - obstacle)
                if distance < 2.0:  # 安全距离
                    obstacle_cost = 100.0 * np.exp(-distance)
                    total_cost += obstacle_cost
                    
    return total_cost
```

**约束设置：**
```python
def _setup_constraints(self, initial_state, target_trajectory, obstacles):
    constraints = []
    
    # 速度约束
    def velocity_constraint(control_flat, initial_state):
        controls = control_flat.reshape(-1, self.control_dim)
        states = self._simulate_trajectory(initial_state, controls)
        velocities = states[:, 2]
        return velocities
        
    constraints.append({
        'type': 'ineq',
        'fun': lambda x: velocity_constraint(x, initial_state) - self.min_velocity
    })
    constraints.append({
        'type': 'ineq',
        'fun': lambda x: self.max_velocity - velocity_constraint(x, initial_state)
    })
    
    # 终端约束（可选）
    if target_trajectory is not None:
        def terminal_constraint(control_flat, initial_state):
            controls = control_flat.reshape(-1, self.control_dim)
            states = self._simulate_trajectory(initial_state, controls)
            terminal_state = states[-1]
            terminal_target = target_trajectory[-1]
            
            position_error = np.linalg.norm(terminal_state[:2] - terminal_target[:2])
            velocity_error = abs(terminal_state[2] - terminal_target[2])
            
            return np.array([-position_error, -velocity_error])
            
        constraints.append({
            'type': 'ineq',
            'fun': lambda x: terminal_constraint(x, initial_state)
        })
        
    return constraints
```

**边界设置：**
```python
def _setup_bounds(self):
    bounds = []
    
    for _ in range(self.horizon):
        # 加速度变化率边界
        bounds.append((self.min_acceleration, self.max_acceleration))
        # 转向角变化率边界
        bounds.append((self.min_steering_rate, self.max_steering_rate))
        
    return bounds
```

#### 2.2.3 优化求解

```python
def optimize(self, initial_state, target_trajectory, obstacles):
    # 初始化决策变量
    initial_controls = np.zeros(self.horizon * self.control_dim)
    
    # 设置约束和边界
    constraints = self._setup_constraints(initial_state, target_trajectory, obstacles)
    bounds = self._setup_bounds()
    
    # 使用SLSQP求解器
    result = minimize(
        self._objective_function,
        initial_controls,
        args=(initial_state, target_trajectory, obstacles),
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
    )
    
    if not result.success:
        print(f"MPC optimization failed: {result.message}")
        
    # 提取优化结果
    optimized_controls = result.x.reshape(-1, self.control_dim)
    optimized_states = self._simulate_trajectory(initial_state, optimized_controls)
    
    return optimized_states, optimized_controls
```

### 2.3 算法特点

**优势：**
- **全局优化**：在预测视野内寻找最优解
- **约束处理**：显式处理各种约束
- **多目标优化**：可同时优化多个目标

**局限性：**
- **计算复杂度高**：随着视野增长计算量急剧增加
- **模型依赖性**：依赖精确的系统模型
- **实时性挑战**：对计算资源要求高

---

## 3. 算法比较分析

### 3.1 性能对比

| 特性 | CBF | MPC |
|------|-----|-----|
| 安全保证 | 理论保证 | 约束保证 |
| 计算复杂度 | 低 | 高 |
| 优化质量 | 局部最优 | 全局最优 |
| 实时性 | 优秀 | 良好 |
| 约束处理 | 安全约束 | 多种约束 |
| 参数调优 | 简单 | 复杂 |

### 3.2 适用场景

**CBF适用场景：**
- 需要强安全保证的应用
- 计算资源有限的系统
- 实时性要求高的场景
- 简单的安全约束

**MPC适用场景：**
- 复杂的多目标优化
- 需要处理多种约束
- 预测视野较长的应用
- 对优化质量要求高的场景

### 3.3 实际测试结果

基于MIND项目的测试结果：

```
成功率：
- MPC: 4/4 (100%)
- CBF: 4/4 (100%)

平均成本：
- MPC: 22,781.0
- CBF: 46,612.6

计算时间：
- 两者都很快（< 1ms）
```

**分析：**
- MPC在成本优化方面表现更好
- CBF更保守但提供了更强的安全保证
- 两者在实时性上都满足要求

---

## 4. 算法改进建议

### 4.1 CBF改进方向

1. **自适应参数调整**
   ```python
   def adaptive_alpha(self, state, obstacles):
       # 根据障碍物距离动态调整α参数
       min_distance = min([np.linalg.norm(state[:2] - obs) for obs in obstacles])
       return self.base_alpha * (1.0 + np.exp(-min_distance/2.0))
   ```

2. **高阶CBF**
   - 考虑系统的高阶动力学
   - 提高控制精度

3. **学习型CBF**
   - 结合机器学习方法
   - 自适应学习安全策略

### 4.2 MPC改进方向

1. **高效求解器**
   - 使用专门的MPC求解器
   - 并行化计算

2. **滚动视野优化**
   ```python
   def adaptive_horizon(self, current_velocity, traffic_density):
       # 根据当前速度和交通密度动态调整视野
       base_horizon = 20
       velocity_factor = current_velocity / 10.0
       density_factor = 1.0 - traffic_density * 0.3
       return int(base_horizon * velocity_factor * density_factor)
   ```

3. **鲁棒MPC**
   - 考虑模型不确定性
   - 提高算法鲁棒性

---

## 5. 总结

CBF和MPC算法各有优势，在MIND项目中：

- **CBF**提供了强大的安全保证，适合需要严格安全约束的场景
- **MPC**在优化质量上表现更好，适合复杂的多目标优化问题

实际应用中，可以考虑将两者结合，形成安全保证下的优化控制策略，以同时满足安全性和性能要求。

---

*文档版本：1.0*  
*最后更新：2025年11月11日*