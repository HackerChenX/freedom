#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复剩余的测试问题

系统性地修复剩余失败测试中的常见问题
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

# 指标名称映射表 - 修复未注册指标问题
INDICATOR_FIXES = {
    # 未注册指标的替代方案
    'ZXM_PATTERN_RECOGNITION': 'ZXM_TECHNICAL_FORM',
    'SPECIAL_INTERFACE': 'COMPOSITE',
    'INTRADAY_VOLATILITY': 'VIX',
    'STOCK_VIX': 'VIX',
    'VIX_INDICATORS': 'VIX',
    'VOLUME_RATIO_INDICATORS': 'VR',
    'UNIFIED_MA_INDICATORS': 'MA',
    
    # 高级分析指标
    'TREND_STRENGTH': 'COMPOSITE',
    'MARKET_SENTIMENT': 'ZXM_MARKET_SENTIMENT',
    'TECHNICAL_ANALYSIS': 'COMPOSITE',
    'ADVANCED_ANALYSIS': 'COMPOSITE',
}

# 参数冲突修复模式
PARAMETER_CONFLICT_PATTERNS = [
    # 移除重复的name参数
    (r"complete_registry\.create_indicator\(\s*'([^']+)',\s*name\s*=\s*[^,)]+,", 
     r"complete_registry.create_indicator('\1',"),
    
    # 移除重复的description参数（如果指标名已经包含描述信息）
    (r"complete_registry\.create_indicator\(\s*'([^']+)',\s*description\s*=\s*[^,)]+,", 
     r"complete_registry.create_indicator('\1',"),
]

def fix_test_issues_in_file(file_path):
    """修复单个文件中的测试问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 1. 修复未注册指标问题
        for old_indicator, new_indicator in INDICATOR_FIXES.items():
            pattern = rf"complete_registry\.create_indicator\('{re.escape(old_indicator)}'"
            replacement = f"complete_registry.create_indicator('{new_indicator}'"
            content = re.sub(pattern, replacement, content)
        
        # 2. 修复参数冲突问题
        for pattern, replacement in PARAMETER_CONFLICT_PATTERNS:
            content = re.sub(pattern, replacement, content)
        
        # 3. 修复特殊的参数冲突情况
        # 移除create_indicator调用中的name参数（当它与第一个位置参数冲突时）
        content = re.sub(
            r"complete_registry\.create_indicator\(\s*'([^']+)',\s*name\s*=\s*[^,)]+",
            r"complete_registry.create_indicator('\1'",
            content
        )
        
        # 4. 修复多行参数冲突
        # 处理跨多行的参数冲突
        lines = content.split('\n')
        in_create_indicator = False
        create_indicator_lines = []
        fixed_lines = []
        
        for line in lines:
            if 'complete_registry.create_indicator(' in line and not line.strip().endswith(')'):
                in_create_indicator = True
                create_indicator_lines = [line]
            elif in_create_indicator:
                create_indicator_lines.append(line)
                if ')' in line:
                    # 处理完整的create_indicator调用
                    full_call = '\n'.join(create_indicator_lines)
                    
                    # 检查是否有name参数冲突
                    if re.search(r"create_indicator\(\s*'[^']+',.*name\s*=", full_call, re.DOTALL):
                        # 移除name参数
                        full_call = re.sub(r',\s*name\s*=\s*[^,)]+', '', full_call)
                    
                    fixed_lines.extend(full_call.split('\n'))
                    in_create_indicator = False
                    create_indicator_lines = []
            else:
                if not in_create_indicator:
                    fixed_lines.append(line)
        
        content = '\n'.join(fixed_lines)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def find_files_with_test_issues():
    """查找所有有测试问题的文件"""
    test_files = []
    tests_dir = Path(root_dir) / "tests" / "unit"
    
    # 重点关注的问题文件
    problem_files = [
        'test_special_interface_indicators.py',
        'test_volatility_and_other_indicators.py',
        'test_advanced_analysis_indicators.py',
        'test_strategy_and_custom_indicators.py',
    ]
    
    for file_name in problem_files:
        file_path = tests_dir / file_name
        if file_path.exists():
            test_files.append(str(file_path))
    
    # 也检查其他可能有问题的文件
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name.startswith("test_") and file_path.name != "__init__.py":
            if str(file_path) not in test_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # 检查是否有需要修复的问题
                    has_issue = False
                    
                    # 检查未注册指标
                    for old_indicator in INDICATOR_FIXES.keys():
                        if f"complete_registry.create_indicator('{old_indicator}'" in content:
                            has_issue = True
                            break
                    
                    # 检查参数冲突
                    if not has_issue:
                        if re.search(r"create_indicator\(\s*'[^']+',.*name\s*=", content):
                            has_issue = True
                    
                    if has_issue:
                        test_files.append(str(file_path))
                        
                except Exception as e:
                    print(f"❌ 无法检查文件 {file_path}: {e}")
    
    return test_files

def main():
    """主函数"""
    print("🚀 开始批量修复剩余测试问题...")
    
    # 查找需要修复的文件
    test_files = find_files_with_test_issues()
    print(f"📊 找到 {len(test_files)} 个需要修复的文件")
    
    if not test_files:
        print("✅ 没有找到需要修复的文件")
        return
    
    # 显示修复映射
    print("\n📋 指标修复映射:")
    for old_indicator, new_indicator in INDICATOR_FIXES.items():
        print(f"  {old_indicator} → {new_indicator}")
    
    # 修复每个文件
    fixed_count = 0
    failed_count = 0
    
    for file_path in test_files:
        print(f"\n🔧 修复文件: {file_path}")
        if fix_test_issues_in_file(file_path):
            print(f"✅ 修复成功")
            fixed_count += 1
        else:
            print(f"❌ 修复失败或无需修复")
            failed_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    
    if fixed_count > 0:
        print("\n🔍 建议运行以下命令验证修复效果:")
        print("python3 -m pytest tests/unit/ -k \"test_calculation_runs_without_error_Mixin or test_returns_dataframe_Mixin\" --tb=line -q")

if __name__ == "__main__":
    main()
