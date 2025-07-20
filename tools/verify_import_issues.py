#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
验证和修复导入问题

系统性地检查和修复测试文件中的导入问题
"""

import os
import re
import sys
from pathlib import Path
import ast
import importlib.util

# 添加项目根目录到路径
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

def check_import_issues(file_path):
    """检查单个文件的导入问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        issues = []
        
        # 检查LogCaptureMixin使用但未导入
        if 'LogCaptureMixin' in content:
            if 'from tests.helper.log_capture import LogCaptureMixin' not in content:
                if 'import LogCaptureMixin' not in content:
                    issues.append('LogCaptureMixin used but not imported')
        
        # 检查MagicMock使用但未导入
        if 'MagicMock' in content:
            if 'from unittest.mock import' not in content or 'MagicMock' not in content:
                # 检查是否有其他形式的导入
                if 'import MagicMock' not in content and 'from unittest.mock import MagicMock' not in content:
                    # 检查是否在from unittest.mock import语句中
                    mock_import_pattern = r'from unittest\.mock import[^\\n]*'
                    mock_imports = re.findall(mock_import_pattern, content)
                    has_magic_mock = any('MagicMock' in imp for imp in mock_imports)
                    if not has_magic_mock:
                        issues.append('MagicMock used but not imported')
        
        # 检查get_config使用但未导入
        if 'get_config(' in content:
            if 'from config.config import get_config' not in content:
                if 'import get_config' not in content:
                    issues.append('get_config used but not imported')
        
        return issues
        
    except Exception as e:
        return [f"Error checking file: {e}"]

def fix_import_issue(file_path, issue_type):
    """修复单个文件的导入问题"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        if issue_type == 'LogCaptureMixin':
            # 检查是否已经有其他log_capture导入
            if 'from tests.helper.log_capture import' in content:
                # 添加LogCaptureMixin到现有导入
                content = re.sub(
                    r'from tests\.helper\.log_capture import ([^\\n]*)',
                    r'from tests.helper.log_capture import \1, LogCaptureMixin',
                    content
                )
            else:
                # 添加新的导入行
                # 找到合适的位置插入导入
                lines = content.split('\\n')
                import_line_added = False
                
                for i, line in enumerate(lines):
                    if line.startswith('from tests.helper') or line.startswith('import'):
                        lines.insert(i + 1, 'from tests.helper.log_capture import LogCaptureMixin')
                        import_line_added = True
                        break
                
                if not import_line_added:
                    # 在文件开头添加导入
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and line.strip():
                            lines.insert(i, 'from tests.helper.log_capture import LogCaptureMixin')
                            break
                
                content = '\\n'.join(lines)
        
        elif issue_type == 'MagicMock':
            # 检查是否已经有unittest.mock导入
            if 'from unittest.mock import' in content:
                # 添加MagicMock到现有导入
                content = re.sub(
                    r'from unittest\.mock import ([^\\n]*)',
                    lambda m: f'from unittest.mock import {m.group(1)}, MagicMock' if 'MagicMock' not in m.group(1) else m.group(0),
                    content
                )
            else:
                # 添加新的导入行
                lines = content.split('\\n')
                import_line_added = False
                
                for i, line in enumerate(lines):
                    if line.startswith('import unittest') or line.startswith('from unittest'):
                        lines.insert(i + 1, 'from unittest.mock import MagicMock')
                        import_line_added = True
                        break
                
                if not import_line_added:
                    # 在文件开头添加导入
                    for i, line in enumerate(lines):
                        if not line.startswith('#') and line.strip():
                            lines.insert(i, 'from unittest.mock import MagicMock')
                            break
                
                content = '\\n'.join(lines)
        
        elif issue_type == 'get_config':
            # 添加get_config导入
            lines = content.split('\\n')
            import_line_added = False
            
            for i, line in enumerate(lines):
                if line.startswith('from config') or line.startswith('import config'):
                    lines.insert(i + 1, 'from config.config import get_config')
                    import_line_added = True
                    break
            
            if not import_line_added:
                # 在文件开头添加导入
                for i, line in enumerate(lines):
                    if not line.startswith('#') and line.strip():
                        lines.insert(i, 'from config.config import get_config')
                        break
                
                content = '\\n'.join(lines)
        
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
    print("🔍 开始验证和修复导入问题...")
    
    # 获取所有测试文件
    tests_dir = Path(root_dir) / "tests"
    test_files = []
    
    for file_path in tests_dir.rglob("*.py"):
        if file_path.name != "__init__.py":
            test_files.append(str(file_path))
    
    print(f"📊 找到 {len(test_files)} 个测试文件")
    
    # 检查导入问题
    files_with_issues = {}
    total_issues = 0
    
    for file_path in test_files:
        issues = check_import_issues(file_path)
        if issues:
            files_with_issues[file_path] = issues
            total_issues += len(issues)
    
    print(f"📋 发现 {total_issues} 个导入问题，涉及 {len(files_with_issues)} 个文件")
    
    if not files_with_issues:
        print("✅ 没有发现导入问题")
        return
    
    # 修复导入问题
    fixed_count = 0
    failed_count = 0
    
    for file_path, issues in files_with_issues.items():
        print(f"\\n🔧 修复文件: {file_path}")
        file_fixed = False
        
        for issue in issues:
            if 'LogCaptureMixin' in issue:
                if fix_import_issue(file_path, 'LogCaptureMixin'):
                    print(f"  ✅ 修复了LogCaptureMixin导入")
                    file_fixed = True
                else:
                    print(f"  ❌ LogCaptureMixin导入修复失败")
            
            elif 'MagicMock' in issue:
                if fix_import_issue(file_path, 'MagicMock'):
                    print(f"  ✅ 修复了MagicMock导入")
                    file_fixed = True
                else:
                    print(f"  ❌ MagicMock导入修复失败")
            
            elif 'get_config' in issue:
                if fix_import_issue(file_path, 'get_config'):
                    print(f"  ✅ 修复了get_config导入")
                    file_fixed = True
                else:
                    print(f"  ❌ get_config导入修复失败")
        
        if file_fixed:
            fixed_count += 1
        else:
            failed_count += 1
    
    print(f"\\n📊 修复完成:")
    print(f"✅ 成功修复: {fixed_count} 个文件")
    print(f"❌ 修复失败: {failed_count} 个文件")
    
    if fixed_count > 0:
        print("\\n🔍 建议运行以下命令验证修复效果:")
        print("python3 tools/test_framework_fixer.py")

if __name__ == "__main__":
    main()
