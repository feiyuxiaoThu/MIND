#!/usr/bin/env python3
"""
MARC uv快速启动脚本

使用uv包管理器快速设置和运行MARC。
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def run_command(cmd, cwd=None, check=True):
    """运行命令"""
    print(f"🔧 执行: {cmd}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, check=check, 
                              capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令失败: {e}")
        if e.stderr:
            print(f"错误: {e.stderr}")
        return False

def check_uv():
    """检查uv是否安装"""
    print("🔍 检查uv...")
    return run_command("uv --version", check=False)

def install_uv():
    """安装uv"""
    print("📦 安装uv...")
    return run_command("pip install uv")

def setup_project():
    """设置项目"""
    project_root = Path(__file__).parent
    
    print("🏗️  设置项目环境...")
    
    # 创建虚拟环境
    if not (project_root / ".venv").exists():
        print("创建虚拟环境...")
        if not run_command("uv venv", cwd=project_root):
            return False
    
    # 激活环境并安装依赖
    print("安装依赖...")
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate && "
    else:
        activate_cmd = "source .venv/bin/activate && "
    
    # 安装基础依赖
    base_deps = [
        "numpy>=1.21.0",
        "scipy>=1.10.0", 
        "matplotlib>=3.5.0",
        "torch>=2.0.0",
        "cvxpy>=1.3.0",
        "pandas>=2.0.0"
    ]
    
    for dep in base_deps:
        if not run_command(f"{activate_cmd} uv pip install {dep}", cwd=project_root):
            print(f"⚠️  {dep} 安装失败，继续...")
    
    return True

def run_marc_demo():
    """运行MARC演示"""
    project_root = Path(__file__).parent
    
    print("🚀 启动MARC演示...")
    
    if sys.platform == "win32":
        activate_cmd = ".venv\\Scripts\\activate && "
    else:
        activate_cmd = "source .venv/bin/activate && "
    
    # 设置Python路径
    env_cmd = f"{activate_cmd} PYTHONPATH={project_root}:$PYTHONPATH"
    
    # 运行演示
    demo_cmd = f"{env_cmd} python run_marc.py --mode demo"
    return run_command(demo_cmd, cwd=project_root, check=False)

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="MARC uv快速启动")
    parser.add_argument("--skip-uv", action="store_true", help="跳过uv安装")
    parser.add_argument("--demo-only", action="store_true", help="仅运行演示")
    
    args = parser.parse_args()
    
    print("🎯 MARC uv快速启动")
    print("=" * 50)
    
    project_root = Path(__file__).parent
    
    # 检查/安装uv
    if not args.skip_uv:
        if not check_uv():
            print("uv未安装，正在安装...")
            if not install_uv():
                print("❌ uv安装失败")
                return 1
        else:
            print("✅ uv已安装")
    
    # 设置项目
    if not args.demo_only:
        if not setup_project():
            print("❌ 项目设置失败")
            return 1
    
    # 运行演示
    if run_marc_demo():
        print("🎉 演示运行成功!")
        return 0
    else:
        print("❌ 演示运行失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
