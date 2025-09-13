#!/usr/bin/python
# -*- coding: UTF-8 -*-

"""
批量修复API路由中的async函数问题
"""

import os
import re
from pathlib import Path

def fix_async_in_file(file_path):
    """修复单个文件中的async问题"""
    print(f"修复文件: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 修复async def为def（除了内部辅助函数）
    content = re.sub(
        r'(@router\.[a-z]+.*\n.*\n.*\n)async def',
        r'\1def',
        content,
        flags=re.MULTILINE
    )
    
    # 修复辅助函数的async def
    content = re.sub(
        r'^async def (_[a-zA-Z_][a-zA-Z0-9_]*)',
        r'def \1',
        content,
        flags=re.MULTILINE
    )
    
    # 修复await调用
    content = re.sub(
        r'await (_[a-zA-Z_][a-zA-Z0-9_]*\()',
        r'\1',
        content
    )
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ 修复完成: {file_path}")

def main():
    """主函数"""
    print("🔧 开始批量修复API路由中的async函数问题...")
    
    # 需要修复的文件列表
    files_to_fix = [
        "api/routers/risk_router.py",
        "api/routers/monitoring_router.py"
    ]
    
    project_root = Path(__file__).parent.parent
    
    for file_path in files_to_fix:
        full_path = project_root / file_path
        if full_path.exists():
            fix_async_in_file(full_path)
        else:
            print(f"⚠️ 文件不存在: {full_path}")
    
    print("🎉 批量修复完成！")

if __name__ == "__main__":
    main()
