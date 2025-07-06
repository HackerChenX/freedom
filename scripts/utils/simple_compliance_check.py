#!/usr/bin/env python3
"""
简单合规性检查脚本
避免复杂的依赖，直接检查代码文件
"""

import os
import re
from pathlib import Path
from collections import defaultdict

def check_compliance():
    """检查项目合规性"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    violations = defaultdict(int)
    total_files = 0
    
    # 检查的目录
    check_dirs = ['utils', 'config', 'db', 'strategy', 'analysis', 'indicators', 'formula', 'scripts', 'bin']
    
    for dir_name in check_dirs:
        target_dir = project_root / dir_name
        if not target_dir.exists():
            continue
            
        for py_file in target_dir.rglob('*.py'):
            # 跳过__pycache__和虚拟环境
            if '__pycache__' in str(py_file) or 'venv' in str(py_file):
                continue
                
            total_files += 1
            
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 检查命名规范违规
                violations['naming'] += check_naming_violations(content)
                
                # 检查代码重复
                violations['duplications'] += check_code_duplications(content)
                
                # 检查导入违规
                violations['imports'] += check_import_violations(content)
                
                # 检查查询违规
                violations['queries'] += check_query_violations(content)
                
            except Exception as e:
                print(f"检查文件失败 {py_file}: {e}")
                
    return violations, total_files

def check_naming_violations(content: str) -> int:
    """检查命名规范违规"""
    violations = 0
    
    # 检查类名（应该是大驼峰）
    class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        if not class_name[0].isupper() or '_' in class_name:
            violations += 1
            
    # 检查函数名（应该是小写+下划线）
    func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        if func_name.startswith('__') and func_name.endswith('__'):
            continue  # 跳过魔法方法
        if not (func_name.islower() or '_' in func_name):
            violations += 1
            
    return violations

def check_code_duplications(content: str) -> int:
    """检查代码重复"""
    violations = 0
    
    # 检查重复的函数定义
    func_names = []
    func_pattern = r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
    for match in re.finditer(func_pattern, content):
        func_name = match.group(1)
        if func_name in func_names:
            violations += 1
        func_names.append(func_name)
        
    # 检查重复的类定义
    class_names = []
    class_pattern = r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*[\(:]'
    for match in re.finditer(class_pattern, content):
        class_name = match.group(1)
        if class_name in class_names:
            violations += 1
        class_names.append(class_name)
        
    return violations

def check_import_violations(content: str) -> int:
    """检查导入违规"""
    violations = 0
    
    # 检查相对导入
    if re.search(r'from\s+\.', content):
        violations += 1
        
    return violations

def check_query_violations(content: str) -> int:
    """检查查询违规"""
    violations = 0
    
    # 检查SELECT code, name, price
    if re.search(r'SELECT\s+\*', content, re.IGNORECASE):
        violations += 1
        
    # 检查没有WHERE条件的查询
    select_pattern = r'SELECT\s+.*?\s+FROM\s+\w+'
    for match in re.finditer(select_pattern, content, re.IGNORECASE | re.DOTALL):
        query = match.group(0)
        if 'WHERE' not in query.upper():
            violations += 1
            
    return violations

def main():
    """主函数"""
    print("开始简单合规性检查...")
    
    violations, total_files = check_compliance()
    
    print(f"\n检查完成！")
    print(f"检查文件数量: {total_files}")
    print(f"违规统计:")
    for violation_type, count in violations.items():
        print(f"  - {violation_type}: {count}")
        
    total_violations = sum(violations.values())
    print(f"\n总违规数量: {total_violations}")
    
    if total_violations < 100:
        print("🎉 恭喜！违规数量已降到100以下，基本达到合规要求！")
    elif total_violations < 500:
        print("👍 不错！违规数量显著减少，继续努力！")
    else:
        print("⚠️ 还需要继续修复违规问题")

if __name__ == '__main__':
    main() 