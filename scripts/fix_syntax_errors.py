#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复语法错误

批量修复测试文件中的特定语法错误
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def fix_syntax_errors_in_file(file_path):
    """修复单个文件中的语法错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复特定的语法错误模式
        patterns = [
            # 修复 \1, LogCaptureMixinn, LogCaptureMixin 错误
            (r'from tests\.helper\.log_capture import \\1, LogCaptureMixinn, LogCaptureMixin',
             'from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin'),
            
            # 修复其他类似的错误
            (r'from tests\.helper\.log_capture import \\1, LogCaptureMixin, LogCaptureMixin',
             'from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin'),
             
            # 修复重复的LogCaptureMixin
            (r'from tests\.helper\.log_capture import ([^,\\n]*), LogCaptureMixin, LogCaptureMixin',
             r'from tests.helper.log_capture import \\1, LogCaptureMixin'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def main():
    """主函数"""
    print("🔧 开始修复语法错误...")
    
    # 获取所有测试文件
    tests_dir = Path(root_dir) / "tests"
    test_files = []
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name != "__init__.py":
            test_files.append(str(file_path))
    
    print(f"📊 找到 {len(test_files)} 个测试文件")
    
    # 修复语法错误
    fixed_count = 0
    
    for file_path in test_files:
        if fix_syntax_errors_in_file(file_path):
            print(f"✅ 修复了文件: {file_path}")
            fixed_count += 1
    
    print(f"\\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")

if __name__ == "__main__":
    main()
