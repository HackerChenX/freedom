#!/usr/bin/env python3
"""
检查技术指标文件语法错误的脚本
识别并修复批量添加get_pattern_info方法后可能产生的语法问题
"""

import os
import ast
import sys
import glob
from typing import List, Tuple

def check_python_syntax(file_path: str) -> Tuple[bool, str]:
    """
    检查Python文件的语法是否正确
    
    Args:
        file_path: 文件路径
        
    Returns:
        Tuple[bool, str]: (是否有语法错误, 错误信息)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 尝试解析AST
        ast.parse(content)
        return False, ""
        
    except SyntaxError as e:
        return True, f"语法错误在第{e.lineno}行: {e.msg}"
    except Exception as e:
        return True, f"其他错误: {e}"

def find_incomplete_methods(file_path: str) -> List[str]:
    """
    查找可能不完整的方法定义
    
    Args:
        file_path: 文件路径
        
    Returns:
        List[str]: 问题列表
    """
    issues = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for i, line in enumerate(lines, 1):
            # 检查方法定义后是否有文档字符串但没有正确结束
            if line.strip().startswith('def ') and ':' in line:
                # 查看接下来的几行
                for j in range(i, min(i + 10, len(lines))):
                    next_line = lines[j].strip()
                    if next_line.startswith('"""') and not next_line.endswith('"""'):
                        # 查找文档字符串的结束
                        doc_end_found = False
                        for k in range(j + 1, min(j + 20, len(lines))):
                            if '"""' in lines[k]:
                                doc_end_found = True
                                break
                        
                        if not doc_end_found:
                            issues.append(f"第{i}行方法可能有未结束的文档字符串")
                    
                    # 检查是否有def紧跟def的情况
                    if j > i and next_line.startswith('def '):
                        prev_method_line = lines[j-1].strip()
                        if not prev_method_line or prev_method_line.startswith('"""'):
                            issues.append(f"第{i}行方法可能没有正确结束，第{j+1}行开始新方法")
                        break
                        
    except Exception as e:
        issues.append(f"检查文件时出错: {e}")
    
    return issues

def main():
    """主函数"""
    print("🔍 开始检查技术指标文件语法错误...")
    
    # 获取所有指标文件
    indicator_patterns = [
        'indicators/*.py',
        'indicators/zxm/*.py',
        'indicators/enhanced/*.py'
    ]
    
    all_files = []
    for pattern in indicator_patterns:
        all_files.extend(glob.glob(pattern))
    
    # 过滤掉__init__.py和一些特殊文件
    indicator_files = [f for f in all_files if not f.endswith('__init__.py') and os.path.isfile(f)]
    
    print(f"📁 找到 {len(indicator_files)} 个指标文件")
    
    syntax_errors = []
    method_issues = []
    
    for file_path in indicator_files:
        print(f"\n🔍 检查文件: {file_path}")
        
        # 检查语法错误
        has_syntax_error, error_msg = check_python_syntax(file_path)
        if has_syntax_error:
            syntax_errors.append((file_path, error_msg))
            print(f"  ❌ 语法错误: {error_msg}")
        else:
            print(f"  ✅ 语法正确")
        
        # 检查方法完整性
        issues = find_incomplete_methods(file_path)
        if issues:
            method_issues.append((file_path, issues))
            for issue in issues:
                print(f"  ⚠️  {issue}")
    
    # 输出总结
    print(f"\n📊 检查结果总结:")
    print(f"  总文件数: {len(indicator_files)}")
    print(f"  语法错误文件数: {len(syntax_errors)}")
    print(f"  方法问题文件数: {len(method_issues)}")
    
    if syntax_errors:
        print(f"\n❌ 发现语法错误的文件:")
        for file_path, error_msg in syntax_errors:
            print(f"  - {file_path}: {error_msg}")
    
    if method_issues:
        print(f"\n⚠️  发现方法问题的文件:")
        for file_path, issues in method_issues:
            print(f"  - {file_path}:")
            for issue in issues:
                print(f"    * {issue}")
    
    if not syntax_errors and not method_issues:
        print(f"\n🎉 所有文件语法检查通过！")
    
    return len(syntax_errors) == 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
