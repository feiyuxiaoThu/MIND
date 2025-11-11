# MARC与MIND规划器运行指南

## 📋 目录
1. [环境准备](#环境准备)
2. [MARC规划器运行](#marc规划器运行)
3. [MIND规划器运行](#mind规划器运行)
4. [测试执行](#测试执行)
5. [性能比较](#性能比较)
6. [故障排除](#故障排除)

## 🛠️ 环境准备

### 系统要求
- Python 3.8+
- Ubuntu/Linux系统
- 至少2GB内存

### 依赖安装
```bash
# 安装基础依赖
pip install numpy scipy matplotlib

# 如果使用原始MIND，还需要：
pip install torch torchvision torchaudio
```

### 快速环境检查
```bash
# 检查Python版本
python3 --version

# 检查依赖包
python3 -c "import numpy, scipy; print('依赖检查通过')"
```

## 🚀 MARC规划器运行

### 1. 基础功能测试
```bash
# 方法1：直接运行测试脚本
python3 -c "
import sys
sys.path.append('.')
from tree_marc.planners.mind_planner import MARCPlanner
import json
import numpy as np

# 加载配置
with open('tree_marc/configs/marc_config.json', 'r') as f:
    config = json.load(f)

# 创建规划器
planner = MARCPlanner(config)

# 测试数据
initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_trajectory = np.array([[10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] for t in range(20)])

# 执行规划
result = planner.plan(initial_state, target_trajectory)

if result['success']:
    print('✓ MARC规划成功!')
    print(f'规划时间: {result[\"planning_time\"]:.3f} 秒')
    print(f'总成本: {result[\"cost\"]:.3f}')
    print(f'轨迹长度: {len(result[\"trajectory\"])}')
else:
    print('✗ 规划失败:', result['reason'])
"
```

### 2. 运行演示程序
```bash
# 进入演示目录
cd tree_marc/examples

# 运行MARC演示
python3 marc_demo.py
```

### 3. 配置文件调整
编辑 `tree_marc/configs/marc_config.json`：
```json
{
  "planner_config": {
    "planning_horizon": 50,
    "dt": 0.1,
    "optimizer_type": "ilqr"
  },
  "risk_config": {
    "alpha": 0.1,
    "max_iterations": 50
  }
}
```

## 🧠 MIND规划器运行

### 1. 模拟MIND规划器测试
```bash
# 运行MIND模拟测试
python3 -c "
import numpy as np
import time

class MockMINDPlanner:
    def plan(self, initial_state, target_trajectory):
        time.sleep(0.1)  # 模拟计算时间
        return {
            'success': True,
            'trajectory': self._simulate_trajectory(initial_state, len(target_trajectory)),
            'planning_time': np.random.uniform(0.05, 0.2),
            'cost': np.random.uniform(100.0, 500.0)
        }
    
    def _simulate_trajectory(self, initial_state, horizon):
        trajectory = np.zeros((horizon + 1, 6))
        trajectory[0] = initial_state
        for t in range(horizon):
            trajectory[t + 1] = [10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0]
        return trajectory

# 测试
mind_planner = MockMINDPlanner()
initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
target_trajectory = np.array([[10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] for t in range(20)])

result = mind_planner.plan(initial_state, target_trajectory)
print('✓ MIND规划成功!')
print(f'规划时间: {result[\"planning_time\"]:.3f} 秒')
"
```

### 2. 原始MIND规划器（需要PyTorch）
```bash
# 安装PyTorch（如果需要）
pip install torch torchvision torchaudio

# 尝试运行原始MIND
python3 -c "
try:
    from planners.mind.planner import MINDPlanner
    print('✓ 原始MIND规划器导入成功')
except ImportError as e:
    print(f'✗ 需要安装依赖: {e}')
"
```

### 3. 使用MIND配置文件
```bash
# 使用现有配置运行
python3 run_sim.py --config configs/demo_1.json
```

## 🧪 测试执行

### 1. MARC单元测试
```bash
# 运行所有MARC测试
python3 tree_marc/tests/test_marc_planner.py

# 预期输出：
# Ran 18 tests in 3.210s
# OK
```

### 2. 测试内容说明
- **MARCPlanner测试**：规划器初始化、基本规划、障碍物处理
- **CVaROptimizer测试**：CVaR优化、敏感性分析
- **BilevelOptimization测试**：双级优化功能
- **TrajectoryTree测试**：轨迹树构建和优化

### 3. 单独运行特定测试
```bash
# 只测试MARC规划器
python3 -m unittest tree_marc.tests.test_marc_planner.TestMARCPlanner

# 只测试CVaR优化器
python3 -m unittest tree_marc.tests.test_marc_planner.TestCVAROptimizer
```

## 📊 性能比较

### 1. 运行MARC vs MIND验证
```bash
# 运行完整验证比较
python3 tree_marc/tests/validation_marc_vs_mind.py

# 输出文件：
# - validation_results.png（性能图表）
# - validation_results.json（详细数据）
```

### 2. 验证报告解读
```
平均规划时间:
  MARC: 0.002 秒
  MIND: 0.119 秒
  比率: 0.02x

成功率:
  MARC: 0.00%
  MIND: 70.00%
  差异: -70.00%
```

### 3. 自定义性能测试
```bash
# 创建自定义测试脚本
cat > custom_test.py << 'EOF'
import time
import numpy as np
from tree_marc.planners.mind_planner import MARCPlanner
import json

# 自定义测试场景
def custom_test():
    config = {"planner_config": {"planning_horizon": 30}}
    planner = MARCPlanner(config)
    
    initial_state = np.array([0.0, 0.0, 8.0, 0.0, 0.0, 0.0])
    target_trajectory = np.array([[8.0 * t * 0.1, 0.0, 8.0, 0.0, 0.0, 0.0] for t in range(30)])
    
    start_time = time.time()
    result = planner.plan(initial_state, target_trajectory)
    end_time = time.time()
    
    print(f"自定义测试 - 规划时间: {end_time - start_time:.3f}秒")
    print(f"结果: {'成功' if result['success'] else '失败'}")

custom_test()
EOF

# 运行自定义测试
python3 custom_test.py
```

## 🔧 故障排除

### 常见问题1：模块导入错误
```bash
# 问题：ModuleNotFoundError: No module named 'numpy'
# 解决：
pip install numpy scipy matplotlib

# 问题：ModuleNotFoundError: No module named 'torch'
# 解决（仅MIND需要）：
pip install torch torchvision torchaudio
```

### 常见问题2：路径问题
```bash
# 确保在正确的目录运行
cd /home/feiyushaw/Documents/Work/e2e/MIND

# 添加Python路径
export PYTHONPATH=/home/feiyushaw/Documents/Work/e2e/MIND:$PYTHONPATH
```

### 常见问题3：权限问题
```bash
# 如果遇到权限错误
chmod +x tree_marc/examples/marc_demo.py
chmod +x tree_marc/tests/test_marc_planner.py
```

### 常见问题4：依赖冲突
```bash
# 创建新的虚拟环境
python3 -m venv test_env
source test_env/bin/activate
pip install numpy scipy matplotlib
# 然后运行测试
```

## 📈 性能优化建议

### 1. MARC优化
```json
// 在 marc_config.json 中调整参数
{
  "planner_config": {
    "max_planning_time": 0.5,  // 减少规划时间
    "planning_horizon": 30      // 减少规划范围
  },
  "risk_config": {
    "max_iterations": 30        // 减少迭代次数
  }
}
```

### 2. 内存优化
```bash
# 监控内存使用
python3 -c "
import psutil
import os
process = psutil.Process(os.getpid())
print(f'内存使用: {process.memory_info().rss / 1024 / 1024:.1f} MB')
"
```

## 📝 日志和调试

### 1. 启用详细日志
```bash
# 设置日志级别
export PYTHONPATH=/home/feiyushaw/Documents/Work/e2e/MIND:$PYTHONPATH
python3 -c "
import logging
logging.basicConfig(level=logging.DEBUG)
from tree_marc.planners.mind_planner import MARCPlanner
# 运行测试...
"
```

### 2. 保存测试结果
```bash
# 创建测试结果目录
mkdir -p test_results

# 运行测试并保存结果
python3 tree_marc/tests/test_marc_planner.py > test_results/unit_test.log 2>&1
python3 tree_marc/tests/validation_marc_vs_mind.py > test_results/validation.log 2>&1
```

## 🎯 快速开始命令汇总

```bash
# 1. 环境检查
python3 --version && python3 -c "import numpy, scipy; print('环境OK')"

# 2. MARC基础测试
python3 -c "
from tree_marc.planners.mind_planner import MARCPlanner
import json, numpy as np
planner = MARCPlanner(json.load(open('tree_marc/configs/marc_config.json')))
result = planner.plan(np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0]), 
                     np.array([[10.0 * t * 0.1, 0.0, 10.0, 0.0, 0.0, 0.0] for t in range(20)]))
print('MARC测试:', '成功' if result['success'] else '失败')
"

# 3. 运行所有测试
python3 tree_marc/tests/test_marc_planner.py

# 4. 性能比较
python3 tree_marc/tests/validation_marc_vs_mind.py
```

## 📞 获取帮助

如果遇到问题：
1. 检查本文档的故障排除部分
2. 查看生成的日志文件
3. 确认依赖包正确安装
4. 验证Python路径设置

---

**注意**：原始MIND规划器需要额外的深度学习依赖，如果只是测试MARC功能，可以跳过相关步骤。