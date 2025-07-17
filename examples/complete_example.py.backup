#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
买点策略反向验证完整示例

演示从买点数据到策略验证的完整流程
"""

import sys
import os
import subprocess
import pandas as pd
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_sample_buypoints():
    """创建示例买点数据"""
    sample_data = {
        'stock_code': ['000001', '600036', '000858', '002415', '600519'],
        'stock_name': ['平安银行', '招商银行', '五粮液', '海康威视', '贵州茅台'],
        'date': ['2024-01-15', '2024-01-16', '2024-01-17', '2024-01-18', '2024-01-19'],
        'price': [12.50, 45.20, 180.30, 35.60, 1680.00],
        'volume': [1000000, 800000, 500000, 1200000, 300000],
        'reason': ['技术突破', '均线金叉', '量价齐升', '突破阻力', '基本面改善']
    }
    
    df = pd.DataFrame(sample_data)
    
    # 确保数据目录存在
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    # 保存示例数据
    sample_file = data_dir / "sample_buypoints.csv"
    df.to_csv(sample_file, index=False, encoding='utf-8-sig')
    
    print(f"✅ 创建示例买点数据: {sample_file}")
    print(f"   包含 {len(df)} 个买点记录")
    
    return str(sample_file)


def run_complete_validation(buypoints_file):
    """运行完整验证流程"""
    print("\n🚀 开始完整验证流程")
    print("=" * 50)
    
    # 设置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = project_root / "results" / f"example_{timestamp}"
    
    # 执行完整验证工作流
    cmd = [
        "python", "scripts/complete_validation_workflow.py",
        "--buypoints", buypoints_file,
        "--output", str(output_dir),
        "--max-conditions", "30",
        "--min-frequency", "2",
        "--top-indicators", "15"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ 验证流程执行成功")
            print("\n📊 执行输出:")
            print(result.stdout)
            
            return str(output_dir)
        else:
            print("❌ 验证流程执行失败")
            print(f"错误信息: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ 执行异常: {e}")
        return None


def analyze_results(output_dir):
    """分析验证结果"""
    if not output_dir:
        return
    
    print("\n📈 分析验证结果")
    print("=" * 50)
    
    output_path = Path(output_dir)
    
    # 查找工作流目录
    workflow_dirs = list(output_path.glob("workflow_*"))
    if not workflow_dirs:
        print("❌ 未找到工作流结果目录")
        return
    
    workflow_dir = workflow_dirs[0]
    
    # 读取摘要报告
    summary_file = workflow_dir / "workflow_summary.txt"
    if summary_file.exists():
        print("📋 工作流摘要:")
        with open(summary_file, 'r', encoding='utf-8') as f:
            print(f.read())
    
    # 读取反向验证报告
    validation_dirs = [
        workflow_dir / "04_reverse_validation" / "original",
        workflow_dir / "04_reverse_validation" / "optimized"
    ]
    
    for i, val_dir in enumerate(validation_dirs):
        strategy_type = "原始策略" if i == 0 else "优化策略"
        
        if val_dir.exists():
            report_files = list(val_dir.glob("reverse_validation_report_*.txt"))
            if report_files:
                print(f"\n📊 {strategy_type}验证结果:")
                print("-" * 30)
                
                with open(report_files[0], 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # 提取关键指标
                    lines = content.split('\n')
                    for line in lines:
                        if any(keyword in line for keyword in ['匹配率:', '精确率:', 'F1分数:', '综合评级:']):
                            print(f"  {line.strip()}")


def demonstrate_individual_steps(buypoints_file):
    """演示各个步骤的单独执行"""
    print("\n🔧 演示各步骤单独执行")
    print("=" * 50)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = project_root / "results" / f"individual_steps_{timestamp}"
    
    steps = [
        {
            "name": "策略生成",
            "cmd": [
                "python", "bin/buypoint_batch_analyzer.py",
                "--input", buypoints_file,
                "--output", str(base_output / "01_analysis")
            ]
        },
        {
            "name": "策略验证",
            "cmd": [
                "python", "scripts/simple_strategy_validator.py",
                "--strategy", str(base_output / "01_analysis" / "generated_strategy.json"),
                "--output", str(base_output / "02_validation")
            ]
        },
        {
            "name": "策略优化",
            "cmd": [
                "python", "scripts/optimize_strategy.py",
                "--strategy", str(base_output / "01_analysis" / "generated_strategy.json"),
                "--output", str(base_output / "03_optimization"),
                "--max-conditions", "25"
            ]
        }
    ]
    
    for step in steps:
        print(f"\n🔄 执行: {step['name']}")
        print(f"命令: {' '.join(step['cmd'])}")
        
        try:
            result = subprocess.run(step['cmd'], cwd=project_root, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ {step['name']}成功")
            else:
                print(f"❌ {step['name']}失败: {result.stderr}")
                
        except Exception as e:
            print(f"❌ {step['name']}异常: {e}")


def show_file_structure(output_dir):
    """显示输出文件结构"""
    if not output_dir:
        return
    
    print("\n📁 输出文件结构")
    print("=" * 50)
    
    output_path = Path(output_dir)
    
    def print_tree(path, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return
        
        if path.is_dir():
            items = sorted(path.iterdir())
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                current_prefix = "└── " if is_last else "├── "
                print(f"{prefix}{current_prefix}{item.name}")
                
                if item.is_dir() and current_depth < max_depth - 1:
                    next_prefix = prefix + ("    " if is_last else "│   ")
                    print_tree(item, next_prefix, max_depth, current_depth + 1)
    
    print_tree(output_path)


def main_completeexample():
    """主函数"""
    print("🎯 买点策略反向验证完整示例")
    print("=" * 60)
    
    # 步骤1: 创建示例数据
    buypoints_file = create_sample_buypoints()
    
    # 步骤2: 运行完整验证
    output_dir = run_complete_validation(buypoints_file)
    
    # 步骤3: 分析结果
    analyze_results(output_dir)
    
    # 步骤4: 显示文件结构
    show_file_structure(output_dir)
    
    # 步骤5: 演示单独步骤
    demonstrate_individual_steps(buypoints_file)
    
    print("\n🎉 示例演示完成！")
    print("\n📚 更多信息:")
    print("  - 详细文档: docs/buypoint_strategy_reverse_validation_guide.md")
    print("  - 快速入门: docs/quick_start_guide.md")
    print("  - 策略验证: docs/strategy_validation_guide.md")
    
    if output_dir:
        print(f"\n📁 示例结果保存在: {output_dir}")


if __name__ == "__main__":
    main_completeexample()
