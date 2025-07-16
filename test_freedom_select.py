#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试 freedom_select.py 的基本功能
"""

import sys
import os
import subprocess
from pathlib import Path

# 添加项目根目录到路径
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))

def test_freedom_select():
    """测试 freedom_select.py 的基本功能"""
    
    freedom_select_path = root_dir / "bin" / "freedom_select.py"
    print(f"🔍 测试文件路径: {freedom_select_path}")
    
    if not freedom_select_path.exists():
        print(f"❌ 文件不存在: {freedom_select_path}")
        return
    
    print("🧪 测试 Freedom Select 工具")
    print("=" * 60)
    
    # 测试1: 显示帮助信息
    print("📋 测试1: 显示帮助信息")
    try:
        result = subprocess.run([
            sys.executable, str(freedom_select_path), "--help"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 帮助信息显示成功")
        else:
            print(f"❌ 帮助信息显示失败: {result.stderr}")
    except Exception as e:
        print(f"❌ 测试异常: {e}")
    
    print()
    
    # 测试2: 列出策略（可能失败，因为需要数据库连接）
    print("📋 测试2: 列出策略")
    try:
        result = subprocess.run([
            sys.executable, str(freedom_select_path), "--list-strategies"
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 策略列表获取成功")
            if result.stdout:
                print(f"输出预览:\n{result.stdout[:200]}...")
        else:
            print(f"⚠️ 策略列表获取失败（可能是数据库连接问题）: {result.stderr}")
    except Exception as e:
        print(f"❌ 测试异常: {e}")
    
    print()
    
    # 测试3: 验证策略文件（创建一个临时策略文件）
    print("📋 测试3: 验证策略文件")
    
    # 创建临时策略文件
    temp_strategy = {
        "strategy_id": "test_strategy",
        "name": "测试策略",
        "description": "用于测试的策略",
        "conditions": [
            {
                "type": "indicator",
                "indicator_id": "MA",
                "period": "daily",
                "operator": ">",
                "value": 0
            }
        ]
    }
    
    import json
    temp_file = root_dir / "test_strategy.json"
    
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(temp_strategy, f, indent=2, ensure_ascii=False)
        
        result = subprocess.run([
            sys.executable, str(freedom_select_path), 
            "--validate-strategy", "--strategy", str(temp_file)
        ], capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print("✅ 策略验证成功")
        else:
            print(f"❌ 策略验证失败: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 测试异常: {e}")
    finally:
        # 清理临时文件
        if temp_file.exists():
            temp_file.unlink()
    
    print()
    print("🎯 测试完成")

if __name__ == "__main__":
    test_freedom_select()