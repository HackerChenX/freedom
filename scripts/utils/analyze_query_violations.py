#!/usr/bin/env python3
"""
查询违规分析脚本
找出具体的查询违规问题
"""

import os
import re
from pathlib import Path

def analyze_query_violations():
    """分析查询违规问题"""
    project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    
    violations = []
    
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
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # 检查SELECT *
                select_star_matches = list(re.finditer(r'SELECT\s+\*', content, re.IGNORECASE))
                for match in select_star_matches:
                    line_num = content[:match.start()].count('\n') + 1
                    violations.append({
                        'file': str(py_file),
                        'line': line_num,
                        'type': 'SELECT_STAR',
                        'content': get_line_content(content, match.start())
                    })
                
                # 检查没有WHERE条件的查询
                select_pattern = r'SELECT\s+.*?\s+FROM\s+\w+'
                for match in re.finditer(select_pattern, content, re.IGNORECASE | re.DOTALL):
                    query = match.group(0)
                    if 'WHERE' not in query.upper() and 'LIMIT' not in query.upper():
                        line_num = content[:match.start()].count('\n') + 1
                        violations.append({
                            'file': str(py_file),
                            'line': line_num,
                            'type': 'NO_WHERE',
                            'content': get_line_content(content, match.start())
                        })
                        
            except Exception as e:
                print(f"分析文件失败 {py_file}: {e}")
                
    return violations

def get_line_content(content: str, position: int) -> str:
    """获取指定位置所在行的内容"""
    lines = content.split('\n')
    line_num = content[:position].count('\n')
    if line_num < len(lines):
        return lines[line_num].strip()
    return ""

def main():
    """主函数"""
    print("开始分析查询违规...")
    
    violations = analyze_query_violations()
    
    print(f"\n找到 {len(violations)} 个查询违规问题：")
    
    # 按类型分组统计
    by_type = {}
    for violation in violations:
        vtype = violation['type']
        if vtype not in by_type:
            by_type[vtype] = []
        by_type[vtype].append(violation)
    
    for vtype, items in by_type.items():
        print(f"\n{vtype}: {len(items)} 个")
        for i, item in enumerate(items[:5]):  # 只显示前5个
            print(f"  {i+1}. {item['file']}:{item['line']} - {item['content'][:80]}...")
        if len(items) > 5:
            print(f"  ... 还有 {len(items) - 5} 个")

if __name__ == '__main__':
    main() 