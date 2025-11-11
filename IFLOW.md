# MIND 项目文档

## 项目概述

MIND (Multi-modal Integrated PredictioN and Decision-making) 是一个用于自动驾驶的多模态集成预测和决策框架。该项目由香港科技大学无人机研究组开发，专门解决在密集和动态环境中导航的挑战，特别是处理多模态交互的复杂性问题。

### 核心特性

- **多模态集成预测和决策**：高效生成覆盖多种交互模态的联合预测和决策
- **自适应交互模态探索**：利用基于学习的场景预测获得具有社会一致性的交互模态
- **模态感知动态分支机制**：生成场景树，有效捕获不同交互模态的演化
- **应急规划**：在交互不确定性下进行应急规划，获得清晰且考虑周全的机动策略

### 技术栈

- **编程语言**：Python 3.10
- **深度学习框架**：PyTorch 2.1.1
- **数据处理**：NumPy, Pandas, PyArrow
- **可视化**：Matplotlib
- **几何处理**：Shapely, PyProj
- **数据集**：Argoverse 2 (av2)
- **优化**：Theano, SciPy
- **仿真**：基于真实世界驾驶数据集的闭环仿真

## 项目结构

```
MIND/
├── agent.py              # 代理类定义
├── ilqr_sim.py           # iLQR仿真相关
├── loader.py             # 数据加载器
├── run_sim.py            # 仿真运行主程序
├── simulator.py          # 仿真器核心实现
├── common/               # 通用工具和模块
│   ├── bbox.py           # 边界框定义
│   ├── data.py           # 数据处理工具
│   ├── geometry.py       # 几何计算工具
│   ├── kinematics.py     # 运动学模型
│   ├── semantic_map.py   # 语义地图处理
│   └── visualization.py  # 可视化工具
├── configs/              # 仿真配置文件
├── data/                 # 数据集存储
├── planners/             # 规划器实现
│   ├── basic/            # 基础规划器
│   ├── ilqr/             # iLQR规划器
│   └── mind/             # MIND规划器核心
│       ├── planner.py    # 主规划器
│       ├── scenario_tree.py  # 场景树生成
│       ├── trajectory_tree.py # 轨迹树优化
│       ├── configs/      # 规划器配置
│       └── networks/     # 神经网络模型
└── misc/                 # 杂项文件（GIF演示等）
```

## 安装和设置

### 环境要求

- Python 3.10
- CUDA支持（可选，用于GPU加速）

### 安装步骤

1. **创建conda环境**
   ```bash
   conda create -n mind python=3.10
   conda activate mind
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

## 运行仿真

### 基本命令

运行闭环仿真：
```bash
python run_sim.py --config configs/demo_{1,2,3,4}.json
```

### 配置说明

配置文件包含以下关键参数：
- `sim_name`: 仿真名称
- `seq_id`: 数据序列ID
- `output_dir`: 输出目录
- `num_threads`: 线程数
- `render`: 是否渲染
- `cl_agents`: 代理列表，包括：
  - `id`: 代理ID
  - `enable_timestep`: 启用时间步
  - `target_velocity`: 目标速度
  - `agent`: 代理类型
  - `planner_config`: 规划器配置路径

### 预期输出

- 整个仿真约需10分钟完成
- 渲染的仿真结果保存在`outputs`文件夹中
- 包含可视化GIF文件展示仿真过程

## 核心组件

### MINDPlanner (planners/mind/planner.py)

主规划器类，负责：
- 设备初始化（CPU/GPU）
- 神经网络初始化和加载
- 场景树生成器初始化
- 轨迹树优化器初始化

### ScenarioTreeGenerator (planners/mind/scenario_tree.py)

场景树生成器，用于：
- 生成多模态交互场景
- 处理交互不确定性
- 构建场景树结构

### TrajectoryTreeOptimizer (planners/mind/trajectory_tree.py)

轨迹树优化器，负责：
- 优化轨迹规划
- 处理约束条件
- 生成最优控制策略

### Simulator (simulator.py)

仿真器主类，提供：
- 仿真环境初始化
- 代理管理
- 仿真循环控制
- 结果渲染

## 开发指南

### 添加新的配置

1. 在`configs/`目录下创建新的JSON配置文件
2. 在`planners/mind/configs/`下创建对应的规划器配置
3. 根据需要调整规划参数和网络配置

### 扩展规划器

1. 继承基础规划器类
2. 实现特定的规划逻辑
3. 更新配置文件以支持新规划器

### 自定义代理

1. 在`agent.py`中扩展代理类
2. 实现特定的观察和控制逻辑
3. 更新加载器以支持新代理类型

## 性能优化

- 使用CUDA加速神经网络计算
- 多线程处理提高仿真效率
- 优化场景树和轨迹树的数据结构

## 故障排除

### 常见问题

1. **CUDA内存不足**：设置`use_cuda: false`使用CPU模式
2. **数据加载失败**：检查数据文件路径和格式
3. **仿真卡顿**：减少线程数或简化场景复杂度

### 调试建议

- 检查配置文件格式和参数
- 验证数据文件完整性
- 使用较小的数据集进行初步测试

## 致谢

本项目感谢以下开源项目的支持：
- [SIMPL](https://github.com/HKUST-Aerial-Robotics/SIMPL)
- [ILQR](https://github.com/anassinator/ilqr)

## 参考文献

相关论文可访问：https://arxiv.org/pdf/2408.13742

项目演示视频：https://www.youtube.com/watch?v=Bwlb5Dz2OZQ