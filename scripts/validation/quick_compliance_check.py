#!/usr/bin/env python3
"""
简化的架构合规检查脚本
"""

import os
import re
from pathlib import Path

def check_compliance():
    """检查架构合规性"""
    violations = 0
    
    # 检查关键违规模式
    violation_patterns = [
        r'from utils\.logger import',
        r'import utils\.logger',
        r'from config import',
        r'import config',
        r'from db\.clickhouse_db import',
        r'from db\.data_manager import'
    ]
    
    for py_file in Path('.').rglob("*.py"):
        if any(skip in str(py_file) for skip in ['__pycache__', '.git', 'venv', 'archive']):
            continue
        
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            for pattern in violation_patterns:
                if re.search(pattern, content):
                    violations += 1
                    print(f"违规: {py_file} - {pattern}")
        
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    
    print(f"\n总违规数: {violations}")
    return violations

if __name__ == "__main__":
    violations = check_compliance()
    exit(1 if violations > 0 else 0)
