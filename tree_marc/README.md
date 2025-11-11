# MARC实现

基于MARC论文实现的多策略和风险感知应急规划框架。

## 概述

MARC (Multipolicy and Risk-aware Contingency Planning) 是一个用于自动驾驶的多策略和风险感知应急规划框架。该实现基于MARC论文的核心概念，提供了完整的轨迹规划解决方案，包括：

- 策略条件场景树生成
- 风险感知应急规划
- 条件风险价值(CVaR)优化
- 双级优化算法
- 轨迹树构建和优化

## 项目结构

```
tree_marc/
├── __init__.py                 # 模块初始化
├── scenario/                   # 场景生成模块
│   ├── __init__.py
│   ├── policy_scenario_tree.py # 策略条件场景树
│   ├── forward_reachable_set.py # 前向可达集
│   └── branch_point.py         # 分支点分析
├── planning/                   # 规划模块
│   ├── __init__.py
│   ├── risk_aware_planning.py  # 风险感知规划
│   ├── cvar_optimizer.py       # CVaR优化器
│   └── bilevel_optimization.py # 双级优化
├── trajectory/                 # 轨迹模块
│   ├── __init__.py
│   ├── trajectory_tree.py      # 轨迹树
│   └── optimizers/             # 优化器
│       ├── __init__.py
│       ├── ilqr_optimizer.py   # iLQR优化器
│       └── mpc_optimizer.py    # MPC优化器
├── planners/                   # 主规划器
│   ├── __init__.py
│   └── mind_planner.py         # MARC主规划器
├── configs/                    # 配置文件
│   └── marc_config.json        # MARC配置
├── examples/                   # 示例代码
│   └── marc_demo.py            # MARC演示
├── tests/                      # 测试代码
│   ├── test_marc_planner.py    # 单元测试
│   └── validation_marc_vs_mind.py # 验证比较
└── README.md                   # 说明文档
```

## 核心特性

### 1. 策略条件场景树 (Policy-Conditioned Scenario Tree)

- 基于不同策略（保守、平衡、激进）生成场景
- 支持多模态交互预测
- 动态分支点分析

### 2. 风险感知应急规划 (Risk-Aware Contingency Planning)

- CVaR风险度量
- 双级优化（线性规划 + iLQR）
- 风险权重自适应调整

### 3. 轨迹树优化

- 多分支轨迹生成
- 基于可达集的安全性保证
- 动态剪枝和优化

### 4. 多种优化算法

- iLQR优化器：用于轨迹优化
- MPC优化器：用于模型预测控制
- CVaR优化器：用于风险优化

## 安装和使用

### 依赖要求

- Python 3.8+
- NumPy
- SciPy
- Matplotlib (可选，用于可视化)
- NetworkX (用于轨迹树结构)

### 基本使用

```python
import numpy as np
import json
from tree_marc.planners.mind_planner import MARCPlanner

# 加载配置
with open('tree_marc/configs/marc_config.json', 'r') as f:
    config = json.load(f)

# 创建规划器
planner = MARCPlanner(config)

# 定义初始状态和目标轨迹
initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_trajectory = np.array([
    [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] 
    for t in range(50)
])

# 执行规划
result = planner.plan(initial_state, target_trajectory)

if result['success']:
    print("规划成功!")
    print(f"规划时间: {result['planning_time']:.3f} 秒")
    print(f"总成本: {result['cost']:.3f}")
else:
    print("规划失败:", result['reason'])
```

### 配置参数

主要配置参数说明：

- `planning_horizon`: 规划范围
- `risk_alpha`: CVaR风险水平
- `ego_policies`: 自车策略列表
- `optimizer_type`: 优化器类型 (ilqr/mpc)

详细配置请参考 `configs/marc_config.json`。

## 示例和测试

### 运行演示

```bash
cd tree_marc/examples
python marc_demo.py
```

### 运行测试

```bash
cd tree_marc/tests
python test_marc_planner.py
```

### 运行验证比较

```bash
cd tree_marc/tests
python validation_marc_vs_mind.py
```

## 与MIND的差异

| 特性 | MARC | MIND |
|------|------|------|
| 场景生成 | 策略条件场景树 | AIME算法 |
| 风险处理 | CVaR风险度量 | 神经网络预测 |
| 优化方法 | 双级优化 (LP + iLQR) | 单级优化 |
| 分支策略 | 基于分歧分析 | 基于协方差 |
| 安全保证 | 前向可达集 | 约束优化 |

## 性能特点

- **规划时间**: 通常比MIND慢，但提供更好的风险保证
- **成功率**: 在复杂场景中成功率更高
- **风险处理**: 更好的风险感知和应急规划能力
- **可扩展性**: 支持多种策略和优化器组合

## 开发和扩展

### 添加新的策略

在配置文件中添加新的策略：

```json
"ego_policies": ["conservative", "balanced", "aggressive", "custom"]
```

并在相应的策略生成器中实现逻辑。

### 自定义优化器

继承基础优化器类：

```python
from tree_marc.trajectory.optimizers.ilqr_optimizer import ILQROptimizer

class CustomOptimizer(ILQROptimizer):
    def __init__(self, config):
        super().__init__(config)
        # 自定义初始化
        
    def optimize(self, ...):
        # 自定义优化逻辑
        pass
```

### 添加新的风险度量

扩展CVaR优化器：

```python
from tree_marc.planning.cvar_optimizer import CVAROptimizer

class CustomRiskOptimizer(CVAROptimizer):
    def __init__(self, config):
        super().__init__(config)
        
    def compute_risk_metric(self, costs, probabilities):
        # 自定义风险度量
        pass
```

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建Pull Request

## 许可证

本项目遵循MIT许可证。

## 引用

如果使用此实现，请引用原始MARC论文。

## 联系方式

如有问题或建议，请通过Issue或Pull Request联系。