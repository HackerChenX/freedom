#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
最终的日志修复脚本

彻底解决所有日志导入问题
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_docstring_import_issue(file_path: Path) -> bool:
    """
    修复文档字符串中的导入问题
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否修复成功
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否有文档字符串中的导入问题
        if '"""\nfrom utils.dependency_injection import get_logger' in content:
            # 修复这种模式
            content = re.sub(
                r'"""\nfrom utils\.dependency_injection import get_logger\n([^"]+)"""\n',
                r'"""\n\1"""\n\nfrom utils.dependency_injection import get_logger\n',
                content
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def ensure_get_logger_import(file_path: Path) -> bool:
    """
    确保文件有正确的 get_logger 导入
    
    Args:
        file_path: 文件路径
        
    Returns:
        bool: 是否修复成功
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 如果文件中使用了 get_logger 但没有导入
        if 'get_logger(' in content and 'from utils.dependency_injection import get_logger' not in content:
            lines = content.split('\n')
            
            # 找到合适的插入位置
            insert_pos = 0
            in_docstring = False
            docstring_quotes = None
            
            for i, line in enumerate(lines):
                stripped = line.strip()
                
                # 跳过文档字符串
                if not in_docstring:
                    if stripped.startswith('"""') or stripped.startswith("'''"):
                        in_docstring = True
                        docstring_quotes = stripped[:3]
                        if stripped.count(docstring_quotes) >= 2:
                            in_docstring = False
                            insert_pos = i + 1
                elif in_docstring:
                    if docstring_quotes in stripped:
                        in_docstring = False
                        insert_pos = i + 1
                        break
                
                # 如果不在文档字符串中，找到第一个导入位置
                if not in_docstring and (stripped.startswith('import ') or stripped.startswith('from ')):
                    insert_pos = i
                    break
                elif not in_docstring and stripped and not stripped.startswith('#'):
                    insert_pos = i
                    break
            
            # 插入导入语句
            lines.insert(insert_pos, 'from utils.dependency_injection import get_logger')
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True
        
        return False
        
    except Exception as e:
        print(f"❌ 修复文件 {file_path} 失败: {e}")
        return False

def fix_all_logger_issues():
    """修复所有日志问题"""
    print("🔧 最终修复所有日志问题...")
    
    # 需要检查的目录
    directories = [
        project_root / "indicators",
        project_root / "analysis",
        project_root / "tests/reverse_validation"
    ]
    
    total_fixed = 0
    
    for directory in directories:
        if not directory.exists():
            continue
        
        print(f"\n处理目录: {directory.relative_to(project_root)}")
        
        for py_file in directory.rglob("*.py"):
            if py_file.name == "__init__.py":
                continue
            
            # 修复文档字符串中的导入问题
            docstring_fixed = fix_docstring_import_issue(py_file)
            
            # 确保有正确的导入
            import_fixed = ensure_get_logger_import(py_file)
            
            if docstring_fixed or import_fixed:
                total_fixed += 1
                print(f"  ✓ {py_file.relative_to(project_root)}")
    
    print(f"\n✅ 总共修复了 {total_fixed} 个文件")
    return total_fixed

def test_critical_imports():
    """测试关键导入"""
    print("\n🔍 测试关键导入...")
    
    critical_modules = [
        'indicators.macd',
        'indicators.rsi',
        'indicators.kdj',
        'indicators.boll',
        'indicators.complete_indicator_registry',
        'tests.reverse_validation.reverse_validation_framework'
    ]
    
    success_count = 0
    
    for module_name in critical_modules:
        try:
            # 重新导入模块
            if module_name in sys.modules:
                del sys.modules[module_name]
            
            module = __import__(module_name, fromlist=[''])
            success_count += 1
            print(f"  ✓ {module_name}")
        except Exception as e:
            print(f"  ✗ {module_name}: {e}")
    
    success_rate = success_count / len(critical_modules)
    print(f"\n关键模块导入成功率: {success_rate:.1%}")
    
    return success_rate > 0.8

def main():
    """主函数"""
    print("=" * 80)
    print("最终日志修复")
    print("=" * 80)
    
    # 修复所有日志问题
    fixed_count = fix_all_logger_issues()
    
    # 测试关键导入
    import_success = test_critical_imports()
    
    print("\n" + "=" * 80)
    print("最终修复总结:")
    print(f"修复文件数: {fixed_count}")
    print(f"关键导入测试: {'✅ 通过' if import_success else '❌ 失败'}")
    print("=" * 80)
    
    return 0 if import_success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
