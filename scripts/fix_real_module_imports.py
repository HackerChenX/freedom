#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复"No module named 'real'"导入错误

查找并修复所有试图导入不存在的"real"模块的指标文件
"""

import os
import re
from pathlib import Path
from typing import List, Dict

def find_files_with_real_imports() -> List[str]:
    """查找包含'real'模块导入的文件"""
    indicators_dir = Path("indicators")
    problem_files = []
    
    # 搜索所有Python文件
    for file_path in indicators_dir.rglob("*.py"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 查找可能的real模块导入
            real_import_patterns = [
                r'from\s+real\s+import',
                r'import\s+real\b',
                r'from\s+\.real\s+import',
                r'from\s+\w+\.real\s+import',
            ]
            
            for pattern in real_import_patterns:
                if re.search(pattern, content):
                    problem_files.append(str(file_path))
                    break
                    
        except Exception as e:
            print(f"⚠️ 无法读取文件 {file_path}: {e}")
    
    return problem_files

def fix_real_import_file(file_path: str) -> bool:
    """修复单个文件的real模块导入问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复各种real模块导入模式
        fixes = [
            # 将 from real import xxx 替换为 from indicators.real_technical_indicators import xxx
            (r'from\s+real\s+import\s+(\w+)', r'from indicators.real_technical_indicators import \1'),
            # 将 import real 替换为 import indicators.real_technical_indicators as real
            (r'import\s+real\b', r'import indicators.real_technical_indicators as real'),
            # 将 from .real import xxx 替换为 from indicators.real_technical_indicators import xxx
            (r'from\s+\.real\s+import\s+(\w+)', r'from indicators.real_technical_indicators import \1'),
            # 将 from xxx.real import yyy 替换为 from indicators.real_technical_indicators import yyy
            (r'from\s+\w+\.real\s+import\s+(\w+)', r'from indicators.real_technical_indicators import \1'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ 修复完成: {file_path}")
            return True
        else:
            print(f"ℹ️ 无需修复: {file_path}")
            return True
            
    except Exception as e:
        print(f"❌ 修复失败 {file_path}: {e}")
        return False

def create_mock_real_modules():
    """创建模拟的real模块以防止导入错误"""
    
    # 创建indicators/real.py模块
    real_module_content = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
模拟real模块

为了兼容性而创建的模拟模块，重定向到real_technical_indicators
"""

# 重新导出real_technical_indicators的内容
from indicators.real_technical_indicators import *
from indicators.real_technical_indicators import RealTechnicalIndicators, RealIndicatorFactory, real_indicator_factory

# 为了向后兼容，提供一些常用的别名
RealIndicators = RealTechnicalIndicators
real_factory = real_indicator_factory
'''
    
    try:
        with open("indicators/real.py", 'w', encoding='utf-8') as f:
            f.write(real_module_content)
        print("✅ 创建模拟real模块: indicators/real.py")
        return True
    except Exception as e:
        print(f"❌ 创建模拟real模块失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始修复'No module named real'导入错误...")
    
    # 创建模拟real模块
    print("\n📦 创建模拟real模块...")
    create_mock_real_modules()
    
    # 查找需要修复的文件
    print("\n🔍 查找包含real模块导入的文件...")
    problem_files = find_files_with_real_imports()
    
    if not problem_files:
        print("✅ 没有发现包含real模块导入的文件")
    else:
        print(f"📋 发现 {len(problem_files)} 个包含real模块导入的文件:")
        for file_path in problem_files:
            print(f"  - {file_path}")
        
        # 修复文件
        print("\n🔧 开始修复文件...")
        success_count = 0
        for file_path in problem_files:
            if fix_real_import_file(file_path):
                success_count += 1
        
        print(f"\n🎉 修复完成: {success_count}/{len(problem_files)} 个文件修复成功")
    
    print("\n✅ 'No module named real'错误修复完成！")

if __name__ == "__main__":
    main()
