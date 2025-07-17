#!/usr/bin/env python3
"""
快速合规性修复脚本
只修复项目内核心文件的关键问题
"""

import os
import sys
import re
from pathlib import Path

# 添加项目根目录到Python路径
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

def fix_project_files():
    """修复项目内的关键文件"""
    project_root = Path(root_dir)
    
    # 要修复的核心目录
    core_dirs = ['utils', 'config', 'db', 'strategy', 'analysis', 'indicators', 'formula']
    
    fixes_count = 0
    
    for dir_name in core_dirs:
        target_dir = project_root / dir_name
        if not target_dir.exists():
            continue
            
        for py_file in target_dir.rglob('*.py'):
            # 跳过__pycache__和虚拟环境
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                original_content = content
                
                # 修复Python标准库名称
                content = fix_standard_library_names(content)
                
                # 修复重复的函数定义
                content = remove_duplicate_functions(content)
                
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixes_count += 1
                    print(f"修复文件: {py_file}")
                    
            except Exception as e:
                print(f"修复文件失败 {py_file}: {e}")
                
    return fixes_count

def fix_standard_library_names(content: str) -> str:
    """修复Python标准库名称"""
    # 修复被错误修改的标准库名称
    fixes = [
        ('User_warning', 'UserWarning'),
        ('Type_error', 'TypeError'),
        ('Value_error', 'ValueError'),
        ('Key_error', 'KeyError'),
        ('Index_error', 'IndexError'),
        ('Attribute_error', 'AttributeError'),
        ('Import_error', 'ImportError'),
        ('Module_not_found_error', 'ModuleNotFoundError'),
        ('File_not_found_error', 'FileNotFoundError'),
        ('Permission_error', 'PermissionError'),
        ('Runtime_error', 'RuntimeError'),
        ('Not_implemented_error', 'NotImplementedError'),
        ('Os_error', 'OSError'),
        ('Io_error', 'IOError'),
        ('Unicode_error', 'UnicodeError'),
        ('Unicode_decode_error', 'UnicodeDecodeError'),
        ('Unicode_encode_error', 'UnicodeEncodeError'),
        ('Cryptography_deprecation_warning', 'CryptographyDeprecationWarning'),
        ('_Finder', '_Finder'),  # 保持原样
        ('Stream_handler', 'StreamHandler'),
        ('Rotating_file_handler', 'RotatingFileHandler'),
        ('get_logger', 'getLogger'),
        ('set_level', 'setLevel'),
        ('add_handler', 'addHandler'),
        ('remove_handler', 'removeHandler'),
        ('set_formatter', 'setFormatter'),
        ('max_bytes', 'maxBytes'),
        ('backup_count', 'backupCount'),
        ('error_Logger', 'error'),
    ]
    
    for wrong_name, correct_name in fixes:
        content = content.replace(wrong_name, correct_name)
        
    return content

def remove_duplicate_functions(content: str) -> str:
    """移除重复的函数定义"""
    lines = content.split('\n')
    seen_functions = set()
    result_lines = []
    skip_until_next_def = False
    
    for line in lines:
        func_match = re.match(r'^(\s*)def\s+(\w+)', line)
        
        if func_match:
            indent, func_name = func_match.groups()
            if func_name in seen_functions:
                skip_until_next_def = True
                continue
            else:
                seen_functions.add(func_name)
                skip_until_next_def = False
        elif skip_until_next_def:
            # 跳过重复函数的内容
            if line.strip() and not line.startswith(' ') and not line.startswith('\t'):
                skip_until_next_def = False
            else:
                continue
                
        if not skip_until_next_def:
            result_lines.append(line)
            
    return '\n'.join(result_lines)

def main_quick_fix_compliance():
    """主函数"""
    print("开始快速修复项目文件...")
    
    fixes_count = fix_project_files()
    
    print(f"\n快速修复完成！")
    print(f"修复文件数量: {fixes_count}")
    print("\n现在可以尝试运行合规性检查")

if __name__ == "__main__":
    main_quick_fix_compliance() 