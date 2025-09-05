#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复导入错误

批量修复测试文件中的导入错误
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def fix_import_errors_in_file(file_path):
    """修复单个文件中的导入错误"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复常见的导入错误
        fixes = [
            # 修复LogCaptureMixin相关的错误导入
            (r'from tests\.helper\.log_capture import.*Log_capture_mixi[^n].*', 
             'from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin'),
            (r'from tests\.helper\.log_capture import.*LogCaptureMixinn.*', 
             'from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin'),
            (r'from tests\.helper\.log_capture import.*Log_capture_mixin, LogCaptureMixin, LogCaptureMixin.*', 
             'from tests.helper.log_capture import Log_capture_mixin, LogCaptureMixin'),
            
            # 修复MagicMock相关的错误导入
            (r'from unittest\.mock import.*Magic_mock.*', 
             'from unittest.mock import MagicMock'),
            
            # 修复重复的导入
            (r'from tests\.helper\.log_capture import ([^,\\n]*), LogCaptureMixin, LogCaptureMixin', 
             r'from tests.helper.log_capture import \\1, LogCaptureMixin'),
        ]
        
        for pattern, replacement in fixes:
            content = re.sub(pattern, replacement, content)
        
        # 确保LogCaptureMixin导入正确
        if 'LogCaptureMixin' in content and 'from tests.helper.log_capture import' in content:
            # 检查是否已经正确导入
            if not re.search(r'from tests\.helper\.log_capture import[^\\n]*LogCaptureMixin', content):
                # 添加LogCaptureMixin到现有导入
                content = re.sub(
                    r'from tests\.helper\.log_capture import ([^\\n]*)',
                    r'from tests.helper.log_capture import \\1, LogCaptureMixin',
                    content
                )
        
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
    print("🔧 开始修复导入错误...")
    
    # 获取所有测试文件
    tests_dir = Path(root_dir) / "tests"
    test_files = []
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name != "__init__.py":
            test_files.append(str(file_path))
    
    print(f"📊 找到 {len(test_files)} 个测试文件")
    
    # 修复导入错误
    fixed_count = 0
    
    for file_path in test_files:
        if fix_import_errors_in_file(file_path):
            print(f"✅ 修复了文件: {file_path}")
            fixed_count += 1
    
    print(f"\\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    
    if fixed_count > 0:
        print("\\n🔍 建议运行以下命令验证修复效果:")
        print("python3 -c \"from tests.helper.log_capture import LogCaptureMixin; print('导入成功')\"")

if __name__ == "__main__":
    main()
