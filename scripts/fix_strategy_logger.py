#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
修复策略模块中的日志问题

批量修复所有策略文件中的 getLogger 问题
"""

import os
import re
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def fix_strategy_logger_issues():
    """修复策略模块中的日志问题"""
    print("🔧 修复策略模块中的日志问题...")
    
    strategy_dir = project_root / "strategy"
    
    if not strategy_dir.exists():
        print("❌ strategy目录不存在")
        return 0
    
    fixed_count = 0
    total_count = 0
    
    for py_file in strategy_dir.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        
        total_count += 1
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 检查是否有 getLogger 问题
            if 'logger = getLogger(__name__)' in content:
                # 添加正确的导入
                if 'from utils.dependency_injection import get_logger' not in content:
                    # 找到合适的插入位置
                    lines = content.split('\n')
                    insert_pos = 0
                    
                    for i, line in enumerate(lines):
                        if line.strip().startswith('from ') or line.strip().startswith('import '):
                            insert_pos = i + 1
                        elif line.strip() and not line.strip().startswith('#'):
                            break
                    
                    lines.insert(insert_pos, 'from utils.dependency_injection import get_logger')
                    content = '\n'.join(lines)
                
                # 替换 getLogger 为 get_logger
                content = content.replace('logger = getLogger(__name__)', 'logger = get_logger(__name__)')
                
                print(f"  ✓ {py_file.relative_to(project_root)}")
                fixed_count += 1
            
            # 写回文件
            if content != original_content:
                with open(py_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                    
        except Exception as e:
            print(f"  ❌ 处理文件 {py_file} 失败: {e}")
    
    print(f"✅ 处理了 {fixed_count}/{total_count} 个策略文件")
    return fixed_count

def main():
    """主函数"""
    print("=" * 80)
    print("修复策略模块日志问题")
    print("=" * 80)
    
    # 修复策略文件
    fixed_count = fix_strategy_logger_issues()
    
    print("\n" + "=" * 80)
    print("修复完成总结:")
    print(f"修复策略文件数: {fixed_count}")
    print("=" * 80)
    
    return 0 if fixed_count >= 0 else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
