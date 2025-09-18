#!/usr/bin/env python3
"""
改进的批量修复缺少等号的语法错误脚本
专门针对真正的赋值语句缺少等号的情况
"""

import re
import os
import sys
from pathlib import Path
import subprocess

def find_missing_equals_patterns(content: str) -> list:
    """
    查找缺少等号的模式 - 改进版本
    """
    patterns = []
    lines = content.split('\n')
    
    for i, line in enumerate(lines, 1):
        # 跳过注释行和空行
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue
            
        # 跳过明显不需要修复的行
        if any(keyword in line for keyword in [
            'class ', 'def ', 'import ', 'from ', 'if ', 'elif ', 'else:', 
            'try:', 'except', 'finally:', 'with ', 'for ', 'while ', 'return ',
            'raise ', 'assert ', 'yield ', 'del ', 'global ', 'nonlocal ',
            'pass', 'break', 'continue'
        ]):
            continue
            
        # 跳过已经有等号的行
        if '=' in line:
            continue
            
        # 跳过装饰器和docstring
        if stripped.startswith('@') or stripped.startswith('"""') or stripped.startswith("'''"):
            continue
            
        # 模式1: variable_name function_call() (缺少等号的函数调用赋值)
        pattern1 = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([a-zA-Z_][a-zA-Z0-9_]*\(.*\))\s*$', line)
        if pattern1:
            indent, var_name, func_call = pattern1.groups()
            patterns.append({
                'line_num': i,
                'original': line,
                'fixed': f"{indent}{var_name} = {func_call}",
                'type': 'function_assignment'
            })
            continue
        
        # 模式2: variable_name expression (缺少等号的表达式赋值)
        # 但要排除复杂的表达式
        pattern2 = re.match(r'^(\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s+([^=\s][^=]*[^=\s])\s*$', line)
        if pattern2:
            indent, var_name, expression = pattern2.groups()
            
            # 确保是赋值语句而不是其他语法
            # 检查常见的赋值表达式模式
            is_assignment = (
                # 算术表达式
                re.search(r'[+\-*/]', expression) or
                # 比较表达式  
                re.search(r'[<>!]', expression) or
                # 逻辑表达式
                re.search(r'\band\b|\bor\b|\bnot\b', expression) or
                # 函数调用
                re.search(r'[a-zA-Z_][a-zA-Z0-9_]*\(', expression) or
                # 索引/属性访问
                re.search(r'[\[\.]', expression) or
                # 字面值
                re.search(r'^\d+\.?\d*$|^[\'"]\w+[\'"]+$|^True$|^False$|^None$', expression.strip()) or
                # 列表/字典
                expression.strip().startswith('[') or expression.strip().startswith('{')
            )
            
            if is_assignment:
                patterns.append({
                    'line_num': i,
                    'original': line,
                    'fixed': f"{indent}{var_name} = {expression}",
                    'type': 'expression_assignment'
                })
                continue
                
        # 模式3: return "string" 应该是 return variable (但只对变量名有效)
        pattern3 = re.match(r'^(\s*)(return\s+)"([a-zA-Z_][a-zA-Z0-9_]*)"(\s*)$', line)
        if pattern3:
            indent, return_stmt, var_name, trailing = pattern3.groups()
            patterns.append({
                'line_num': i,
                'original': line,
                'fixed': f"{indent}{return_stmt}{var_name}{trailing}",
                'type': 'return_string_fix'
            })
            continue
            
        # 模式4: 类别名定义 (最后一行常见)
        pattern4 = re.match(r'^([A-Z_][A-Z0-9_]*)\s+([A-Z][a-zA-Z0-9_]*)$', line.strip())
        if pattern4:
            class_alias, class_name = pattern4.groups()
            patterns.append({
                'line_num': i,
                'original': line,
                'fixed': line.replace(f"{class_alias} {class_name}", f"{class_alias} = {class_name}"),
                'type': 'class_alias'
            })
    
    return patterns

def fix_file(file_path: str, dry_run: bool = True) -> tuple:
    """
    修复单个文件
    返回 (是否有修改, 修改数量, 错误信息)
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        patterns = find_missing_equals_patterns(content)
        
        if not patterns:
            return False, 0, None
            
        if dry_run:
            print(f"\n📁 文件: {file_path}")
            print(f"🔍 发现 {len(patterns)} 个需要修复的模式:")
            for p in patterns:
                print(f"  第{p['line_num']}行 ({p['type']}):")
                print(f"    原始: {repr(p['original'])}")
                print(f"    修复: {repr(p['fixed'])}")
            return True, len(patterns), None
        else:
            # 实际修复
            lines = content.split('\n')
            for pattern in reversed(patterns):  # 从后往前修复，避免行号变化
                lines[pattern['line_num'] - 1] = pattern['fixed']
            
            # 写入文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            
            return True, len(patterns), None
            
    except Exception as e:
        return False, 0, str(e)

def test_syntax(file_path: str) -> tuple:
    """
    测试文件语法
    返回 (是否通过, 错误信息)
    """
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'py_compile', file_path],
            capture_output=True,
            text=True,
            cwd=os.path.dirname(file_path) or '.'
        )
        return result.returncode == 0, result.stderr if result.returncode != 0 else None
    except Exception as e:
        return False, str(e)

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='改进的批量修复缺少等号的语法错误')
    parser.add_argument('paths', nargs='+', help='要处理的文件或目录路径')
    parser.add_argument('--dry-run', action='store_true', help='只预览，不实际修改')
    parser.add_argument('--test-syntax', action='store_true', help='修复后测试语法')
    parser.add_argument('--pattern', default='*.py', help='文件模式 (默认: *.py)')
    
    args = parser.parse_args()
    
    # 收集所有Python文件
    files_to_process = []
    for path_str in args.paths:
        path = Path(path_str)
        if path.is_file() and path.suffix == '.py':
            files_to_process.append(str(path))
        elif path.is_dir():
            files_to_process.extend(str(p) for p in path.rglob(args.pattern))
    
    if not files_to_process:
        print("❌ 没有找到要处理的Python文件")
        return
    
    print(f"🎯 开始处理 {len(files_to_process)} 个文件...")
    
    total_fixes = 0
    files_with_issues = 0
    files_with_errors = 0
    
    for file_path in files_to_process:
        has_changes, fix_count, error = fix_file(file_path, dry_run=args.dry_run)
        
        if error:
            print(f"❌ 处理文件失败: {file_path}")
            print(f"   错误: {error}")
            files_with_errors += 1
        elif has_changes:
            files_with_issues += 1
            total_fixes += fix_count
            
            if not args.dry_run:
                print(f"✅ 修复完成: {file_path} ({fix_count} 个修复)")
                
                # 测试语法
                if args.test_syntax:
                    syntax_ok, syntax_error = test_syntax(file_path)
                    if syntax_ok:
                        print(f"   ✅ 语法检查通过")
                    else:
                        print(f"   ❌ 语法检查失败: {syntax_error}")
    
    # 统计结果
    print(f"\n📊 处理结果:")
    print(f"   总文件数: {len(files_to_process)}")
    print(f"   需要修复的文件: {files_with_issues}")
    print(f"   总修复数量: {total_fixes}")
    print(f"   处理错误: {files_with_errors}")
    
    if args.dry_run:
        print(f"\n💡 这是预览模式，没有实际修改文件")
        print(f"   要执行实际修复，请移除 --dry-run 参数")

if __name__ == '__main__':
    main()
