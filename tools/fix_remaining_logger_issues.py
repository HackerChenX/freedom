#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复剩余的日志问题

专门修复 logging.get_logger 和其他日志相关问题
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_logging_get_logger_issues():
    """修复 logging.get_logger 问题"""
    print("🔧 修复 logging.get_logger 问题...")
    
    indicators_dir = project_root / "indicators"
    analysis_dir = project_root / "analysis"
    
    fixed_files = 0
    
    # 搜索所有Python文件
    for directory in [indicators_dir, analysis_dir]:
        if not directory.exists():
            continue
            
        for py_file in directory.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                original_content = content
                
                # 修复 logging.get_logger
                if 'logging.get_logger' in content:
                    content = content.replace('logging.get_logger', 'get_logger')
                    print(f"  修复 {py_file.relative_to(project_root)}: logging.get_logger -> get_logger")
                
                # 确保有正确的导入
                if 'get_logger(' in content and 'from utils.dependency_injection import get_logger' not in content:
                    # 在导入部分添加正确的导入
                    lines = content.split('\n')
                    import_added = False
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('from ') or line.strip().startswith('import '):
                            continue
                        elif line.strip() and not line.strip().startswith('#'):
                            # 在第一个非导入、非注释行之前插入导入
                            lines.insert(i, 'from utils.dependency_injection import get_logger')
                            import_added = True
                            break
                    
                    if import_added:
                        content = '\n'.join(lines)
                        print(f"  添加导入 {py_file.relative_to(project_root)}: from utils.dependency_injection import get_logger")
                
                # 如果内容有变化，写回文件
                if content != original_content:
                    with open(py_file, 'w', encoding='utf-8') as f:
                        f.write(content)
                    fixed_files += 1
                    
            except Exception as e:
                print(f"  ❌ 处理文件 {py_file} 失败: {e}")
    
    print(f"✅ 修复了 {fixed_files} 个文件的日志问题")
    return fixed_files

def fix_buypoint_analyzer():
    """修复买点分析器的日志问题"""
    print("🔧 修复买点分析器...")
    
    buypoint_file = project_root / "analysis/buypoints/analyze_buypoints.py"
    
    if not buypoint_file.exists():
        print("  ❌ 买点分析器文件不存在")
        return False
    
    try:
        with open(buypoint_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有 getLogger 问题
        if 'getLogger(' in content and 'get_logger(' not in content:
            content = content.replace('getLogger(', 'get_logger(')
            print("  修复 getLogger -> get_logger")
        
        # 确保有正确的导入
        if 'from utils.dependency_injection import get_logger' not in content:
            # 在现有的 get_logger 导入行替换
            content = re.sub(
                r'from utils\.dependency_injection import getLogger',
                'from utils.dependency_injection import get_logger',
                content
            )
            print("  修复导入语句")
        
        with open(buypoint_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("  ✅ 买点分析器修复完成")
        return True
        
    except Exception as e:
        print(f"  ❌ 修复买点分析器失败: {e}")
        return False

def fix_pattern_registry_singleton():
    """修复形态注册表单例问题"""
    print("🔧 修复形态注册表单例问题...")
    
    registry_file = project_root / "indicators/pattern_registry.py"
    
    if not registry_file.exists():
        print("  ❌ 形态注册表文件不存在")
        return False
    
    try:
        with open(registry_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否需要修复 get_pattern_registry 函数
        if 'def get_pattern_registry()' in content:
            # 确保返回的是实例而不是类
            pattern = r'def get_pattern_registry\(\):[^}]+?return PatternRegistry\(\)'
            if re.search(pattern, content, re.DOTALL):
                print("  ✅ get_pattern_registry 已经返回实例")
            else:
                # 查找并修复
                lines = content.split('\n')
                in_function = False
                for i, line in enumerate(lines):
                    if 'def get_pattern_registry()' in line:
                        in_function = True
                    elif in_function and line.strip().startswith('return'):
                        if 'PatternRegistry()' not in line:
                            lines[i] = line.replace('PatternRegistry', 'PatternRegistry()')
                            print("  修复 get_pattern_registry 返回实例")
                        break
                
                content = '\n'.join(lines)
        
        with open(registry_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("  ✅ 形态注册表修复完成")
        return True
        
    except Exception as e:
        print(f"  ❌ 修复形态注册表失败: {e}")
        return False

def main():
    """主函数"""
    print("=" * 80)
    print("修复剩余的日志和组件问题")
    print("=" * 80)
    
    # 修复日志问题
    fixed_files = fix_logging_get_logger_issues()
    
    # 修复买点分析器
    buypoint_fixed = fix_buypoint_analyzer()
    
    # 修复形态注册表
    registry_fixed = fix_pattern_registry_singleton()
    
    print("\n" + "=" * 80)
    print("修复完成总结:")
    print(f"修复日志文件数: {fixed_files}")
    print(f"买点分析器: {'✅ 修复成功' if buypoint_fixed else '❌ 修复失败'}")
    print(f"形态注册表: {'✅ 修复成功' if registry_fixed else '❌ 修复失败'}")
    print("=" * 80)
    
    return 0 if (fixed_files >= 0 and buypoint_fixed and registry_fixed) else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
