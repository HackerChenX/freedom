#!/usr/bin/env python3
"""
指标重复检测脚本
检测indicators目录下的重复实现
"""

import os
import re
from pathlib import Path
from typing import Dict, List

def detect_duplicate_indicators():
    """检测重复指标"""
    indicators_dir = Path('indicators')
    indicator_names = {}
    duplicates = {}

    if indicators_dir.exists():
        for py_file in indicators_dir.rglob('*.py'):
            if py_file.name not in ['__init__.py', 'base_indicator.py']:
                indicator_name = extract_indicator_name(py_file)
                if indicator_name:
                    if indicator_name in indicator_names:
                        if indicator_name not in duplicates:
                            duplicates[indicator_name] = [indicator_names[indicator_name]]
                        duplicates[indicator_name].append(str(py_file))
                    else:
                        indicator_names[indicator_name] = str(py_file)

    return duplicates

def extract_indicator_name(file_path: Path) -> str:
    """提取指标名称"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 提取类名中的指标名称
        match = re.search(r'class\s+(\w*([A-Z][a-z]+)\w*)', content)
        if match:
            return match.group(2).upper()
    except Exception:
        pass

    return ""

if __name__ == "__main__":
    duplicates = detect_duplicate_indicators()
    if duplicates:
        print("发现重复指标:")
        for name, files in duplicates.items():
            print(f"  {name}: {len(files)}个文件")
            for file in files:
                print(f"    - {file}")
    else:
        print("未发现重复指标")
