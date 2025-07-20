#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统性修复所有指标文件中的日志导入问题

修复 getLogger vs get_logger 不一致问题，确保88+个技术指标能够正确加载
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Dict, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_logger_import_issues() -> List[Tuple[str, List[str]]]:
    """
    查找所有有日志导入问题的文件
    
    Returns:
        List[Tuple[str, List[str]]]: (文件路径, 问题列表)
    """
    issues = []
    
    # 搜索indicators目录下的所有Python文件
    indicators_dir = project_root / "indicators"
    
    if not indicators_dir.exists():
        print(f"❌ indicators目录不存在: {indicators_dir}")
        return issues
    
    for py_file in indicators_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
            
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            file_issues = []
            
            # 检查各种日志导入问题
            for i, line in enumerate(lines, 1):
                # 问题1: 使用 getLogger 而不是 get_logger
                if 'getLogger(' in line and 'get_logger(' not in line:
                    file_issues.append(f"Line {i}: 使用了 getLogger 而不是 get_logger")
                
                # 问题2: 错误的导入语句
                if 'from utils.logger import' in line:
                    file_issues.append(f"Line {i}: 错误的导入 utils.logger")
                
                # 问题3: 直接导入 logging.getLogger
                if 'logging.getLogger(' in line and 'import logging' in content:
                    file_issues.append(f"Line {i}: 直接使用 logging.getLogger")
                
                # 问题4: 缺少正确的导入
                if 'get_logger(' in line and 'from utils.dependency_injection import get_logger' not in content:
                    file_issues.append(f"Line {i}: 使用了 get_logger 但未正确导入")
            
            if file_issues:
                issues.append((str(py_file), file_issues))
                
        except Exception as e:
            print(f"❌ 无法检查文件 {py_file}: {e}")
    
    return issues

def fix_logger_imports_in_file(file_path: str) -> bool:
    """
    修复单个文件中的日志导入问题
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否成功修复
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        modified = False
        new_lines = []
        has_correct_import = False
        import_section_end = 0
        
        # 第一遍：找到导入部分的结束位置，检查是否已有正确导入
        for i, line in enumerate(lines):
            if line.strip().startswith('from ') or line.strip().startswith('import '):
                import_section_end = i
                if 'from utils.dependency_injection import get_logger' in line:
                    has_correct_import = True
            elif line.strip() and not line.strip().startswith('#'):
                break
        
        # 第二遍：修复问题
        for i, line in enumerate(lines):
            new_line = line
            
            # 修复1: 替换 getLogger 为 get_logger
            if 'getLogger(' in line and 'get_logger(' not in line:
                new_line = re.sub(r'getLogger\(', 'get_logger(', new_line)
                modified = True
                print(f"  修复 getLogger -> get_logger: Line {i+1}")
            
            # 修复2: 移除错误的导入
            if 'from utils.logger import' in line:
                new_line = f"# {line}  # 已修复：错误的导入"
                modified = True
                print(f"  注释错误导入: Line {i+1}")
            
            # 修复3: 替换直接的 logging.getLogger
            if 'logging.getLogger(' in line:
                new_line = re.sub(r'logging\.getLogger\(', 'get_logger(', new_line)
                modified = True
                print(f"  修复 logging.getLogger -> get_logger: Line {i+1}")
            
            new_lines.append(new_line)
        
        # 添加正确的导入（如果还没有）
        if not has_correct_import and modified:
            # 在导入部分结束后添加正确的导入
            insert_pos = import_section_end + 1
            new_lines.insert(insert_pos, "from utils.dependency_injection import get_logger")
            modified = True
            print(f"  添加正确的导入: from utils.dependency_injection import get_logger")
        
        # 写回文件
        if modified:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(new_lines))
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def fix_all_logger_imports() -> Dict[str, int]:
    """
    修复所有文件中的日志导入问题
    
    Returns:
        Dict[str, int]: 修复统计信息
    """
    print("🔍 查找日志导入问题...")
    issues = find_logger_import_issues()
    
    if not issues:
        print("✅ 未发现日志导入问题")
        return {"total_files": 0, "fixed_files": 0, "failed_files": 0}
    
    print(f"📋 发现 {len(issues)} 个文件有日志导入问题")
    
    stats = {"total_files": len(issues), "fixed_files": 0, "failed_files": 0}
    
    for file_path, file_issues in issues:
        print(f"\n🔧 修复文件: {os.path.relpath(file_path, project_root)}")
        for issue in file_issues:
            print(f"  - {issue}")
        
        if fix_logger_imports_in_file(file_path):
            stats["fixed_files"] += 1
            print(f"  ✅ 修复成功")
        else:
            stats["failed_files"] += 1
            print(f"  ❌ 修复失败")
    
    return stats

def verify_fixes() -> bool:
    """
    验证修复结果
    
    Returns:
        bool: 是否所有问题都已修复
    """
    print("\n🔍 验证修复结果...")
    remaining_issues = find_logger_import_issues()
    
    if not remaining_issues:
        print("✅ 所有日志导入问题已修复")
        return True
    else:
        print(f"❌ 仍有 {len(remaining_issues)} 个文件存在问题:")
        for file_path, file_issues in remaining_issues:
            print(f"  {os.path.relpath(file_path, project_root)}: {len(file_issues)} 个问题")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("系统性修复日志导入问题")
    print("=" * 80)
    
    # 修复所有问题
    stats = fix_all_logger_imports()
    
    # 显示统计信息
    print("\n" + "=" * 80)
    print("修复统计:")
    print(f"总文件数: {stats['total_files']}")
    print(f"成功修复: {stats['fixed_files']}")
    print(f"修复失败: {stats['failed_files']}")
    
    if stats['total_files'] > 0:
        success_rate = stats['fixed_files'] / stats['total_files']
        print(f"成功率: {success_rate:.1%}")
    
    # 验证修复结果
    if verify_fixes():
        print("\n🎉 所有日志导入问题已成功修复！")
        return 0
    else:
        print("\n⚠️  部分问题未能修复，需要手动检查")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
