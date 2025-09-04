#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
批量修复语法错误

修复批量修复脚本引入的语法错误
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
        
        # 修复 ])    def 模式 - 在 ]) 和 def 之间添加换行符
        content = re.sub(r'\]\)\s*def\s+', ']\)\n\n    def ', content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def find_files_with_syntax_errors():
    """查找所有有语法错误的测试文件"""
    test_files = []
    tests_dir = Path(root_dir) / "tests" / "unit"
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name.startswith("test_") and file_path.name != "__init__.py":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 检查是否有语法错误模式
                has_issue = re.search(r'\]\)\s*def\s+', content)
                
                if has_issue:
                    test_files.append(str(file_path))
                    
            except Exception as e:
                print(f"❌ 无法检查文件 {file_path}: {e}")
    
    return test_files

def main():
    """主函数"""
    print("🚀 开始批量修复语法错误...")
    
    # 查找需要修复的文件
    test_files = find_files_with_syntax_errors()
    print(f"📊 找到 {len(test_files)} 个需要修复的文件")
    
    if not test_files:
        print("✅ 没有找到需要修复的文件")
        return
    
    # 修复每个文件
    fixed_count = 0
    failed_count = 0
    
    for file_path in test_files:
        print(f"\n🔧 修复文件: {file_path}")
        if fix_syntax_errors_in_file(file_path):
            print(f"✅ 修复成功")
            fixed_count += 1
        else:
            print(f"❌ 修复失败")
            failed_count += 1
    
    print(f"\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")

if __name__ == "__main__":
    main()
