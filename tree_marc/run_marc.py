#!/usr/bin/env python3
"""
MARC 运行脚本

提供MARC规划器的快速启动和演示功能。
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def setup_environment():
    """设置环境"""
    print("🔧 设置MARC运行环境...")
    
    # 检查必要的目录
    required_dirs = [
        "tree_marc",
        "tree_marc/configs",
        "tree_marc/tests",
        "tree_marc/examples"
    ]
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            print(f"❌ 缺少必要目录: {dir_path}")
            return False
    
    print("✅ 环境检查通过")
    return True

def run_demo():
    """运行演示"""
    print("🚀 启动MARC演示...")
    
    try:
        from tree_marc.examples.marc_demo import main as demo_main
        demo_main()
        return True
    except Exception as e:
        print(f"❌ 演示运行失败: {e}")
        return False

def run_tests():
    """运行测试"""
    print("🧪 运行MARC测试...")
    
    try:
        import subprocess
        result = subprocess.run([
            sys.executable, "-m", "pytest", 
            "tree_marc/tests/", 
            "-v", "--tb=short"
        ], cwd=project_root, capture_output=True, text=True)
        
        print(result.stdout)
        if result.stderr:
            print("错误信息:")
            print(result.stderr)
        
        return result.returncode == 0
    except Exception as e:
        print(f"❌ 测试运行失败: {e}")
        return False

def run_validation():
    """运行验证"""
    print("📊 运行MARC vs MIND验证...")
    
    try:
        from tree_marc.tests.validation_marc_vs_mind import main as validation_main
        validation_main()
        return True
    except Exception as e:
        print(f"❌ 验证运行失败: {e}")
        return False

def check_dependencies():
    """检查依赖"""
    print("📦 检查依赖包...")
    
    required_packages = [
        "numpy",
        "scipy", 
        "matplotlib",
        "torch",
        "cvxpy"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} (未安装)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  缺少依赖包: {', '.join(missing_packages)}")
        print("请使用以下命令安装依赖:")
        print("uv pip install " + " ".join(missing_packages))
        return False
    
    print("✅ 所有依赖检查通过")
    return True

def create_simple_scenario():
    """创建简单测试场景"""
    print("🎯 创建简单测试场景...")
    
    try:
        from tree_marc.planners.mind_planner import MARCPlanner, PlanningState
        
        # 初始化规划器
        config_path = project_root / "tree_marc" / "configs" / "marc_config.json"
        planner = MARCPlanner(str(config_path))
        
        # 创建简单场景
        initial_state = np.array([0.0, 0.0, 10.0, 0.0, 0.0, 0.0])
        target_lane = np.array([
            [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [10.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [20.0, 0.0, 10.0, 0.0, 0.0, 0.0],
            [30.0, 0.0, 10.0, 0.0, 0.0, 0.0]
        ])
        
        # 执行规划
        planning_state = planner.update_planning_state(initial_state, target_lane)
        result = planner.plan(planning_state)
        
        if result.success:
            print(f"✅ 规划成功!")
            print(f"   - 成本: {result.cost:.2f}")
            print(f"   - 风险值: {result.risk_value:.2f}")
            print(f"   - 计算时间: {result.computation_time*1000:.2f} ms")
            return True
        else:
            print("❌ 规划失败")
            return False
            
    except Exception as e:
        print(f"❌ 场景创建失败: {e}")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MARC规划器运行脚本")
    parser.add_argument("--mode", choices=["demo", "test", "validation", "check", "simple"], 
                       default="demo", help="运行模式")
    parser.add_argument("--skip-deps", action="store_true", help="跳过依赖检查")
    
    args = parser.parse_args()
    
    print("🎯 MARC (Multipolicy and Risk-aware Contingency Planning)")
    print("=" * 60)
    
    # 环境检查
    if not setup_environment():
        sys.exit(1)
    
    # 依赖检查
    if not args.skip_deps:
        if not check_dependencies():
            print("\n💡 解决方案:")
            print("1. 确保已安装uv: pip install uv")
            print("2. 安装依赖: uv pip install -e .")
            print("3. 或跳过依赖检查: python run_marc.py --skip-deps")
            sys.exit(1)
    
    # 根据模式执行
    success = False
    
    if args.mode == "demo":
        success = run_demo()
    elif args.mode == "test":
        success = run_tests()
    elif args.mode == "validation":
        success = run_validation()
    elif args.mode == "check":
        print("✅ 环境检查完成")
        success = True
    elif args.mode == "simple":
        success = create_simple_scenario()
    
    # 输出结果
    print("\n" + "=" * 60)
    if success:
        print("🎉 运行成功!")
        sys.exit(0)
    else:
        print("❌ 运行失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()